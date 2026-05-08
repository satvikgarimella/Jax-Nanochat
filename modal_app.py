import modal
import re

app = modal.App("nanochat")
volume = modal.Volume.from_name("nanochat-model", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "jax[cuda12]", "tiktoken", "numpy", "fastapi[standard]", "ddgs"
)

@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
    volumes={"/model": volume},
)
class NanoChat:
    @modal.enter()
    def load_model(self):
        import pickle
        import os
        
        print("=== CONTAINER STARTING ===")
        print(f"Files in /model: {os.listdir('/model')}")
        
        print("Loading pickle file...")
        with open("/model/nanochat_large_instruct.pkl", "rb") as f:
            ckpt = pickle.load(f)
        print("Pickle loaded successfully!")
        
        # Store checkpoint, initialize JAX later on first request
        self.ckpt = ckpt
        self.initialized = False
        print("Ready to handle requests!")
    
    def _ensure_initialized(self):
        if self.initialized:
            return
        
        print("Starting initialization...")
        import jax
        print("JAX imported")
        import jax.numpy as jnp
        print("jax.numpy imported")
        import tiktoken
        print("tiktoken imported")
        
        print("Converting params to JAX arrays...")
        self.params = jax.tree.map(jnp.array, self.ckpt["params"])
        print("Params converted")
        
        self.cfg = self.ckpt["config"]
        self.enc = tiktoken.get_encoding("gpt2")
        
        print("Setting up model config...")
        block_size = self.cfg["block_size"]
        n_head = self.cfg["n_head"]
        n_embd = self.cfg["n_embd"]
        head_dim = n_embd // n_head
        
        print("Creating mask and RoPE...")
        self.mask = jnp.where(jnp.tril(jnp.ones((block_size, block_size))), 0.0, -jnp.inf)
        dims = jnp.arange(0, head_dim, 2)
        freqs = 1.0 / (10000.0 ** (dims / head_dim))
        pos = jnp.arange(block_size)
        angles = jnp.repeat(jnp.outer(pos, freqs), 2, axis=-1)
        self.cos, self.sin = jnp.cos(angles), jnp.sin(angles)
        self.block_size = block_size
        self.n_head = n_head
        
        print("Creating JIT-compiled forward pass and sampling step...")
        from functools import partial
        
        @partial(jax.jit, static_argnames=["n_head", "top_k"])
        def jitted_step(tokens, params, n_head, mask, cos, sin, seq_len, temperature, top_p, top_k, key):
            def rms_norm(x, gamma, eps=1e-6):
                ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
                return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
            
            def apply_rope(xq, xk, cos, sin):
                def rotate_half(x):
                    x1, x2 = jnp.split(x, 2, axis=-1)
                    return jnp.concatenate([-x2, x1], axis=-1)
                c = cos[None, None, :, :].astype(xq.dtype)
                s = sin[None, None, :, :].astype(xq.dtype)
                return (xq * c) + (rotate_half(xq) * s), (xk * c) + (rotate_half(xk) * s)
            
            def attention(x, attn, n_head, mask, cos, sin):
                b, seq, n_embd = x.shape
                hd = n_embd // n_head
                q = jnp.dot(x, attn["wq"]).reshape(b, seq, n_head, hd).transpose(0, 2, 1, 3)
                k = jnp.dot(x, attn["wk"]).reshape(b, seq, n_head, hd).transpose(0, 2, 1, 3)
                v = jnp.dot(x, attn["wv"]).reshape(b, seq, n_head, hd).transpose(0, 2, 1, 3)
                q, k = apply_rope(q, k, cos[:seq], sin[:seq])
                scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(hd) + mask[:seq, :seq]
                w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
                return jnp.dot(jnp.matmul(w, v).transpose(0, 2, 1, 3).reshape(b, seq, -1), attn["wo"])
            
            def block(x, p, n_head, mask, cos, sin):
                h = rms_norm(x, p["norm1"])
                x = x + attention(h, p["attn"], n_head, mask, cos, sin)
                h = rms_norm(x, p["norm2"])
                return x + jnp.dot(jax.nn.gelu(jnp.dot(h, p["mlp"]["w1"])), p["mlp"]["w2"])
            
            x = params["token_emb"][tokens].astype(jnp.bfloat16)
            for bp in params["blocks"]:
                x = block(x, bp, n_head, mask, cos, sin)
            logits = jnp.dot(rms_norm(x, params["final_norm"]), params["output_head"]).astype(jnp.float32)
            
            # Extract logit for the next token
            next_logits = logits[0, seq_len - 1, :]
            
            # Apply temperature
            next_logits = next_logits / jnp.maximum(temperature, 1e-8)

            if top_k > 0:
                top_values = jax.lax.top_k(next_logits, top_k)[0]
                threshold = top_values[-1]
                next_logits = jnp.where(next_logits < threshold, -jnp.inf, next_logits)
            
            # Top-P filtering
            sorted_indices = jnp.argsort(-next_logits, axis=-1)
            sorted_logits = jnp.take_along_axis(next_logits, sorted_indices, axis=-1)
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            
            # Shift cumulative probs to ensure we keep the first token that crosses threshold
            shifted_cumulative = cumulative_probs - sorted_probs
            sorted_mask = shifted_cumulative < top_p
            sorted_logits = jnp.where(sorted_mask, sorted_logits, -jnp.inf)
            
            unsort_indices = jnp.argsort(sorted_indices, axis=-1)
            filtered_logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)
            
            next_token = jax.random.categorical(key, filtered_logits, axis=-1)
            return next_token
        
        self._jitted_step = jitted_step
        
        print("Warming up with test generation...")
        self._generate("hi", max_tokens=5)
        self.initialized = True
        print("Initialization complete!")
    
    def _generate(self, prompt, max_tokens=100, temperature=0.7, top_p=0.85, top_k=40):
        import jax
        import jax.numpy as jnp
        import numpy as np
        
        tokens = self.enc.encode(prompt)
        key = jax.random.PRNGKey(np.random.randint(0, 10000))
        
        for _ in range(max_tokens):
            if len(tokens) >= self.block_size:
                context = tokens[-self.block_size:]
            else:
                context = tokens
            
            seq_len = len(context)
            padded_tokens = context + [0] * (self.block_size - seq_len)
            tokens_arr = jnp.array([padded_tokens])
            
            key, subkey = jax.random.split(key)
            
            next_token = self._jitted_step(
                tokens_arr, self.params, self.n_head, self.mask, self.cos, self.sin,
                seq_len, temperature, top_p, top_k, subkey
            )
            
            tokens.append(int(next_token))
            
            decoded = self.enc.decode(tokens)
            if "User:" in decoded.split("Assistant:")[-1]:
                break
        
        return self.enc.decode(tokens)

    def _extract_response(self, output: str) -> str:
        if "Assistant:" in output:
            response = output.split("Assistant:")[-1]
            if "User:" in response:
                response = response.split("User:")[0]
            return response.strip()
        return output.strip()

    def _question_text(self, message: str) -> str:
        if "Question:" in message:
            return message.split("Question:", 1)[-1].split("\n", 1)[0].strip()
        return message.strip()

    def _is_factual_prompt(self, message: str) -> bool:
        lowered = message.lower()
        return (
            "background:" in lowered
            or "question:" in lowered
            or lowered.startswith(("what ", "who ", "when ", "where ", "why ", "how ", "explain ", "define "))
        )

    def _is_question_echo(self, candidate: str, message: str) -> bool:
        candidate_text = candidate.strip()
        if candidate_text.endswith("?"):
            return True

        question_text = self._question_text(message).lower().rstrip("?.! ")
        candidate_lower = candidate_text.lower()
        if candidate_lower.startswith(("question:", "background:", "user:", "assistant:")):
            return True

        normalized_candidate = " ".join(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\\-]*", candidate_lower))
        normalized_question = " ".join(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\\-]*", question_text))
        return normalized_candidate == normalized_question

    def _clean_response(self, message: str, response: str) -> str:
        text = re.sub(r"\s+", " ", response).strip()
        if not text:
            return text

        if not self._is_factual_prompt(message):
            return text

        text = re.sub(r"^(answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        cleaned: list[str] = []
        seen: set[str] = set()

        for sentence in sentences:
            candidate = sentence.strip()
            if not candidate:
                continue
            if self._is_question_echo(candidate, message):
                continue
            if candidate.lower().startswith(("background:", "question:", "user:", "assistant:")):
                continue

            key = re.sub(r"\W+", "", candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(candidate)
            if len(cleaned) == 2:
                break

        return " ".join(cleaned) if cleaned else text

    def _looks_unstable_response(self, message: str, response: str) -> bool:
        stripped = response.strip()
        if len(stripped) < 12:
            return True

        lowered = stripped.lower()
        words = re.findall(r"[a-zA-Z']+", lowered)
        if len(words) >= 18 and len(set(words)) / len(words) < 0.55:
            return True

        if self._is_factual_prompt(message):
            question_text = self._question_text(message)
            terms = [
                term for term in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", question_text.lower())
                if term not in {"what", "when", "where", "who", "does", "about", "tell", "explain", "define"}
            ]
            if terms and not any(term in lowered for term in terms):
                return True

        return False
    
    def _format_chat_prompt(self, message: str, history: list[dict] | None = None) -> str:
        prompt_parts = []

        for turn in (history or [])[-8:]:
            role = turn.get("role")
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")

        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def _chat(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        history: list[dict] | None = None,
    ) -> str:
        self._ensure_initialized()
        temperature = max(0.2, min(float(temperature), 0.85))
        max_tokens = max(1, min(int(max_tokens), 160))

        prompt = self._format_chat_prompt(message, history or [])
        output = self._generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.85,
            top_k=40,
        )
        response = self._clean_response(message, self._extract_response(output))

        if self._looks_unstable_response(message, response):
            retry_output = self._generate(
                prompt,
                max_tokens=min(max_tokens, 96),
                temperature=min(temperature, 0.35),
                top_p=0.7,
                top_k=20,
            )
            retry_response = self._clean_response(message, self._extract_response(retry_output))
            if retry_response and not self._looks_unstable_response(message, retry_response):
                return retry_response

        return response

    @modal.method()
    def chat(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        history: list[dict] | None = None,
    ) -> str:
        return self._chat(message, temperature, max_tokens, history)

    @modal.fastapi_endpoint(method="POST")
    def api(self, request: dict) -> dict:
        try:
            print(f"Received request: {request}")
            message = request.get("message", "")
            history = request.get("history", [])
            temperature = request.get("temperature", 0.7)
            max_tokens = request.get("max_tokens", 100)
            
            response = self._chat(message, temperature, max_tokens, history)
            print(f"Got response: {response[:100]}...")
            return {"response": response}
        except Exception as e:
            print(f"ERROR in API: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# upload model to volume
@app.function(volumes={"/model": volume})
def upload_model(local_path: str):
    import shutil
    shutil.copy(local_path, "/model/nanochat_large_final.pkl")
    volume.commit()
    print("model uploaded")


@app.function(volumes={"/model": volume})
def list_volume():
    import os
    files = os.listdir("/model")
    print("Files in volume:", files)
    return files


@app.local_entrypoint()
def main():
    print("Listing volume contents:")
    files = list_volume.remote()
    print(files)
    # model = NanoChat()
    # print(model.chat.remote("What is machine learning?"))
