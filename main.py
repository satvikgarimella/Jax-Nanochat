"""
NanoChat - Instruction-Following AI Assistant
Trained on Alpaca dataset with tiktoken BPE tokenizer.
"""

import jax
import jax.numpy as jnp
import pickle
import time
import os
import tiktoken
from dataclasses import dataclass

from model import forward
from generate import generate
from utils import create_causal_mask, precompute_rope


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class Config:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    dtype: str = "float32"


SYSTEM_PROMPT = "You are NanoChat, a friendly and helpful AI assistant."


# ============================================================================
# Model Loading
# ============================================================================

def load_checkpoint(path: str):
    """Load model checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = jax.tree.map(jnp.array, checkpoint['params'])
    cfg = checkpoint.get('config', {})
    step = checkpoint.get('step', 0)
    
    print(f"Loaded checkpoint from step {step}")
    return params, cfg


# ============================================================================
# Chat Functions
# ============================================================================

def format_prompt(user_message: str, history: list = None, style: str = "casual") -> str:
    """Format prompt with system message and history.
    
    Styles:
        - casual: User: ... Assistant: ... (natural conversation)
        - formal: ### Instruction: ... ### Response: ... (Alpaca style)
        - simple: Q: ... A: ... (Q&A style)
    """
    if style == "casual":
        prompt = ""
        if history:
            for msg in history[-4:]:
                if msg['role'] == 'user':
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
        prompt += f"User: {user_message}\nAssistant:"
    
    elif style == "formal":
        prompt = f"### System: {SYSTEM_PROMPT}\n"
        if history:
            for msg in history[-4:]:
                if msg['role'] == 'user':
                    prompt += f"### Instruction: {msg['content']}\n"
                else:
                    prompt += f"### Response: {msg['content']}\n"
        prompt += f"### Instruction: {user_message}\n### Response:"
    
    else:  # simple
        prompt = f"Q: {user_message}\nA:"
    
    return prompt


def extract_response(text: str) -> str:
    """Extract just the response from generated text."""
    # Try different format markers
    if "Assistant:" in text:
        response = text.split("Assistant:")[-1]
    elif "### Response:" in text:
        response = text.split("### Response:")[-1]
    elif "A:" in text:
        response = text.split("A:")[-1]
    else:
        response = text
    
    # Stop at next turn marker
    for marker in ["User:", "### Instruction:", "### System:", "Q:", "\n\n\n"]:
        if marker in response:
            response = response.split(marker)[0]
    
    return response.strip()


def warmup(params, config, mask, cos, sin, enc):
    """Warmup JIT compilation."""
    print("Warming up model (JIT compilation)...", end=" ", flush=True)
    start = time.perf_counter()
    
    dummy_tokens = enc.encode("Hello")
    _ = generate(
        params, dummy_tokens, config, mask, cos, sin,
        max_new_tokens=5, temperature=0.8, top_p=0.9,
        key=jax.random.PRNGKey(0)
    )
    
    print(f"done in {time.perf_counter() - start:.1f}s")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 50)
    print("NanoChat - Instruction-Following Assistant")
    print("=" * 50)
    
    # Load tiktoken
    print("\nLoading tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    # Find checkpoint (prefer fine-tuned large model)
    ckpt_paths = [
        "nanochat_large_final.pkl",        # Fine-tuned 162M param model
        "nanochat_large_pretrained.pkl",   # Pretrained 162M param model
        "nanochat_alpaca_final.pkl",       # 10M param model
        "checkpoints/nanochat_large_final.pkl",
        "checkpoints/nanochat_large_pretrained.pkl",
        "checkpoints/nanochat_alpaca_final.pkl",
    ]
    
    ckpt_path = None
    for path in ckpt_paths:
        if os.path.exists(path):
            ckpt_path = path
            break
    
    if not ckpt_path:
        print("ERROR: No checkpoint found!")
        print("Run: modal volume get nanochat-checkpoints nanochat_alpaca_final.pkl ./checkpoints/")
        return
    
    # Load model
    print(f"Loading model from {ckpt_path}...")
    params, cfg = load_checkpoint(ckpt_path)
    
    config = Config(
        block_size=cfg.get('block_size', 512),
        vocab_size=cfg.get('vocab_size', 50257),
        n_layer=cfg.get('n_layer', 6),
        n_head=cfg.get('n_head', 6),
        n_embd=cfg.get('n_embd', 384),
    )
    
    print(f"  Model: {config.n_layer}L-{config.n_head}H-{config.n_embd}D")
    print(f"  Vocab: {config.vocab_size:,} tokens (GPT-2 BPE)")
    print(f"  Context: {config.block_size} tokens")
    
    # Precompute
    mask = create_causal_mask(config.block_size)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rope(config.block_size, head_dim)
    
    # Warmup
    warmup(params, config, mask, cos, sin, enc)
    
    # Chat interface
    print("\n" + "=" * 50)
    print("Chat with NanoChat!")
    print("Commands: /clear, /temp X, /topp X, /tokens X, /style [casual|formal|simple], /quit")
    print("=" * 50)
    print(f"\nSystem: {SYSTEM_PROMPT}\n")
    
    history = []
    temperature = 0.7
    top_p = 0.9
    max_tokens = 50  # Start low for CPU
    style = "casual"  # casual, formal, or simple
    key = jax.random.PRNGKey(42)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break
            elif user_input.lower() == "/clear":
                history = []
                print("History cleared.\n")
                continue
            elif user_input.lower().startswith("/temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature = {temperature}\n")
                except:
                    print("Usage: /temp 0.7\n")
                continue
            elif user_input.lower().startswith("/topp "):
                try:
                    top_p = float(user_input.split()[1])
                    print(f"Top-p = {top_p}\n")
                except:
                    print("Usage: /topp 0.9\n")
                continue
            elif user_input.lower().startswith("/style "):
                new_style = user_input.split()[1].lower()
                if new_style in ["casual", "formal", "simple"]:
                    style = new_style
                    print(f"Style = {style}\n")
                else:
                    print("Usage: /style casual|formal|simple\n")
                continue
            elif user_input.lower().startswith("/tokens "):
                try:
                    max_tokens = int(user_input.split()[1])
                    print(f"Max tokens = {max_tokens}\n")
                except:
                    print("Usage: /tokens 50\n")
                continue
            
            # Format prompt
            prompt = format_prompt(user_input, history, style=style)
            prompt_tokens = enc.encode(prompt)
            
            # Check context length
            if len(prompt_tokens) > config.block_size - 100:
                print("(Context too long, clearing old messages)")
                history = history[-2:] if len(history) > 2 else []
                prompt = format_prompt(user_input, history, style=style)
                prompt_tokens = enc.encode(prompt)
            
            # Generate
            key, subkey = jax.random.split(key)
            start = time.perf_counter()
            
            generated_tokens = generate(
                params, prompt_tokens, config, mask, cos, sin,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                key=subkey
            )
            
            elapsed = time.perf_counter() - start
            
            # Decode and extract response
            full_text = enc.decode(generated_tokens)
            response = extract_response(full_text)
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Print response
            new_tokens = len(generated_tokens) - len(prompt_tokens)
            print(f"\nNanoChat: {response}")
            print(f"[{new_tokens} tokens in {elapsed:.2f}s]\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
