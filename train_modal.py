"""
NanoChat Training on Modal (Cloud GPU)
Run with: modal run train_modal.py
"""

import modal

# Create Modal app
app = modal.App("nanochat-training")

# Define the image with all dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "jax[cuda12]",
    "jaxlib",
    "optax",
    "numpy",
    "tiktoken",
    "datasets",  # For loading FineWeb
    "huggingface_hub",
)

# Create a volume to persist checkpoints
volume = modal.Volume.from_name("nanochat-checkpoints", create_if_missing=True)


# ============================================================================
# SCALED UP MODEL - 85M params
# ============================================================================
@app.function(
    image=image,
    gpu="A100",  # A100 for larger model
    timeout=43200,  # 12 hours
    volumes={"/checkpoints": volume},
)
def train_large(steps: int = 100000):
    """Train a larger 85M parameter model on FineWeb-Edu."""
    import os
    import pickle
    import time
    import json
    
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    import tiktoken
    from datasets import load_dataset
    
    print("=" * 60)
    print("NanoChat LARGE - 85M Parameter Model")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    
    # ========== Configuration - SCALED UP ==========
    class Config:
        block_size = 1024      # Longer context
        vocab_size = 50257     # GPT-2
        n_layer = 12           # 12 layers (was 6)
        n_head = 12            # 12 heads (was 6)
        n_embd = 768           # 768 dim (was 384)
        dropout = 0.0
        dtype = "bfloat16"
        
        batch_size = 8         # Smaller batch for larger model
        learning_rate = 3e-4
        max_steps = steps
        warmup_steps = 2000
        weight_decay = 0.1
        grad_clip = 1.0
        
        log_interval = 100
        save_interval = 5000
    
    config = Config()
    
    # ========== Data Loading - FineWeb-Edu ==========
    print("\n[1/4] Loading FineWeb-Edu dataset...")
    
    # Load a sample of FineWeb-Edu (education-focused web text)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample
        split="train",
        streaming=True
    )
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize ~50M tokens for training
    print("  Tokenizing (this takes a few minutes)...")
    all_tokens = []
    target_tokens = 50_000_000  # 50M tokens
    
    for i, example in enumerate(dataset):
        tokens = enc.encode(example['text'])
        all_tokens.extend(tokens)
        
        if len(all_tokens) >= target_tokens:
            break
        
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i} docs, {len(all_tokens):,} tokens...")
    
    all_tokens = np.array(all_tokens[:target_tokens], dtype=np.int32)
    split_idx = int(len(all_tokens) * 0.99)
    train_data = all_tokens[:split_idx]
    val_data = all_tokens[split_idx:]
    
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")
    
    # ========== Model ==========
    print("\n[2/4] Initializing 85M parameter model...")
    
    def rms_norm(x, gamma, eps=1e-6):
        ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
    
    def apply_rope(xq, xk, cos, sin):
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)
        cos = cos[None, None, :, :].astype(xq.dtype)
        sin = sin[None, None, :, :].astype(xq.dtype)
        return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)
    
    def self_attention(x, params, n_head, mask, cos, sin):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // n_head
        
        q = jnp.dot(x, params['wq'])
        k = jnp.dot(x, params['wk'])
        v = jnp.dot(x, params['wv'])
        
        q = q.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        
        q, k = apply_rope(q, k, cos, sin)
        
        scale = jnp.sqrt(jnp.array(head_dim, dtype=x.dtype))
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
        scores = scores + mask.astype(scores.dtype)
        
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        output = jnp.matmul(weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        
        return jnp.dot(output, params['wo'])
    
    def transformer_block(x, params, n_head, mask, cos, sin):
        h = rms_norm(x, params['norm1'])
        x = x + self_attention(h, params['attn'], n_head, mask, cos, sin)
        h = rms_norm(x, params['norm2'])
        x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, params['mlp']['w1'])), params['mlp']['w2'])
        return x
    
    def forward(params, tokens, n_head, mask, cos, sin):
        x = params['token_emb'][tokens].astype(jnp.bfloat16)
        for block_params in params['blocks']:
            x = transformer_block(x, block_params, n_head, mask, cos, sin)
        x = rms_norm(x, params['final_norm'])
        return jnp.dot(x, params['output_head']).astype(jnp.float32)
    
    # Initialize params
    key = jax.random.PRNGKey(42)
    std = 0.02
    
    def init_weight(k, shape, scale=1.0):
        return jax.random.normal(k, shape) * std * scale
    
    k1, k2, k3 = jax.random.split(key, 3)
    residual_scale = 1.0 / np.sqrt(2 * config.n_layer)
    
    params = {
        'token_emb': init_weight(k1, (config.vocab_size, config.n_embd)) * 0.5,
        'blocks': [
            {
                'attn': {
                    'wq': init_weight(jax.random.fold_in(k2, i*4+0), (config.n_embd, config.n_embd)),
                    'wk': init_weight(jax.random.fold_in(k2, i*4+1), (config.n_embd, config.n_embd)),
                    'wv': init_weight(jax.random.fold_in(k2, i*4+2), (config.n_embd, config.n_embd)),
                    'wo': init_weight(jax.random.fold_in(k2, i*4+3), (config.n_embd, config.n_embd), residual_scale),
                },
                'mlp': {
                    'w1': init_weight(jax.random.fold_in(k2, i*2+100), (config.n_embd, config.n_embd * 4)),
                    'w2': init_weight(jax.random.fold_in(k2, i*2+101), (config.n_embd * 4, config.n_embd), residual_scale),
                },
                'norm1': jnp.ones((config.n_embd,)),
                'norm2': jnp.ones((config.n_embd,)),
            }
            for i in range(config.n_layer)
        ],
        'final_norm': jnp.ones((config.n_embd,)),
        'output_head': init_weight(k3, (config.n_embd, config.vocab_size))
    }
    
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {num_params:,}")  # Should be ~85M
    
    # Precompute
    def create_mask(seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -jnp.inf)
    
    def precompute_rope(seq_len, head_dim, theta=10000.0):
        dims = jnp.arange(0, head_dim, 2)
        freqs = 1.0 / (theta ** (dims / head_dim))
        pos = jnp.arange(seq_len)
        angles = jnp.outer(pos, freqs)
        angles = jnp.repeat(angles, 2, axis=-1)
        return jnp.cos(angles), jnp.sin(angles)
    
    mask = create_mask(config.block_size)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rope(config.block_size, head_dim)
    
    # ========== Optimizer ==========
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=1e-5
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    opt_state = optimizer.init(params)
    
    # ========== Training ==========
    @jax.jit
    def train_step(params, opt_state, tokens, targets):
        def loss_fn(p):
            logits = forward(p, tokens, config.n_head, mask, cos, sin)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(target_log_probs)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    print("\n[3/4] Training...")
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for step in range(config.max_steps):
        max_start = len(train_data) - config.block_size - 1
        starts = rng.integers(0, max_start, size=(config.batch_size,))
        tokens = np.stack([train_data[i:i + config.block_size] for i in starts])
        targets = np.stack([train_data[i + 1:i + 1 + config.block_size] for i in starts])
        
        params, opt_state, loss = train_step(params, opt_state, jnp.array(tokens), jnp.array(targets))
        
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * config.batch_size * config.block_size / max(elapsed, 1)
            print(f"Step {step:6d} | Loss {float(loss):.4f} | {tok_per_sec:.0f} tok/s")
        
        if step > 0 and step % config.save_interval == 0:
            ckpt = {
                'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
                'step': step,
                'config': {
                    'block_size': config.block_size,
                    'vocab_size': config.vocab_size,
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_embd': config.n_embd,
                }
            }
            with open(f"/checkpoints/nanochat_large_step_{step}.pkl", 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"  Saved checkpoint")
            volume.commit()
    
    # Final save
    print("\n[4/4] Saving final model...")
    ckpt = {
        'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
        'step': config.max_steps,
        'config': {
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
        }
    }
    with open(f"/checkpoints/nanochat_large_pretrained.pkl", 'wb') as f:
        pickle.dump(ckpt, f)
    volume.commit()
    
    total_time = time.time() - start_time
    print(f"\nPretraining complete! Time: {total_time/3600:.1f} hours")
    print(f"Final loss: {float(loss):.4f}")
    print("Download with: modal volume get nanochat-checkpoints nanochat_large_pretrained.pkl")
    
    return float(loss)


# ============================================================================
# FINE-TUNE LARGE MODEL on diverse instructions
# ============================================================================
@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={"/checkpoints": volume},
)
def finetune_large(steps: int = 30000):
    """Fine-tune the large pretrained model on diverse instructions."""
    import os
    import pickle
    import time
    import json
    import urllib.request
    
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    import tiktoken
    
    print("=" * 60)
    print("Fine-tuning 85M Model on Diverse Instructions")
    print("=" * 60)
    
    # Load pretrained model
    ckpt_path = "/checkpoints/nanochat_large_pretrained.pkl"
    print(f"Loading pretrained model from {ckpt_path}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    params = jax.tree.map(jnp.array, ckpt['params'])
    cfg = ckpt['config']
    print(f"  Config: {cfg}")
    
    # Load Alpaca dataset
    print("\nLoading Alpaca dataset...")
    alpaca_url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    alpaca_path = "/tmp/alpaca.json"
    urllib.request.urlretrieve(alpaca_url, alpaca_path)
    
    with open(alpaca_path, 'r') as f:
        alpaca_data = json.load(f)
    
    print(f"  Loaded {len(alpaca_data)} examples")
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Create diverse prompt variations to handle casual inputs
    def format_example(ex, style='formal'):
        instruction = ex.get('instruction', '')
        inp = ex.get('input', '')
        output = ex.get('output', '')
        
        if style == 'formal':
            prompt = f"### Instruction: {instruction}\n"
            if inp:
                prompt += f"### Input: {inp}\n"
            prompt += f"### Response: {output}\n\n"
        elif style == 'casual':
            # Casual variations
            prompt = f"User: {instruction}"
            if inp:
                prompt += f" {inp}"
            prompt += f"\nAssistant: {output}\n\n"
        elif style == 'simple':
            # Simple Q&A
            prompt = f"Q: {instruction}\nA: {output}\n\n"
        
        return prompt
    
    # Create training data with mixed styles
    all_tokens = []
    for ex in alpaca_data:
        # Add each example in multiple formats
        for style in ['formal', 'casual', 'simple']:
            tokens = enc.encode(format_example(ex, style))
            all_tokens.extend(tokens)
    
    train_data = np.array(all_tokens, dtype=np.int32)
    print(f"  Training tokens: {len(train_data):,}")
    
    # Config
    block_size = cfg['block_size']
    n_head = cfg['n_head']
    n_embd = cfg['n_embd']
    n_layer = cfg['n_layer']
    batch_size = 8
    
    # Model functions (same as pretraining)
    def rms_norm(x, gamma, eps=1e-6):
        ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
    
    def apply_rope(xq, xk, cos, sin):
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)
        cos = cos[None, None, :, :].astype(xq.dtype)
        sin = sin[None, None, :, :].astype(xq.dtype)
        return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)
    
    def self_attention(x, attn_params, n_head, mask, cos, sin):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // n_head
        q = jnp.dot(x, attn_params['wq']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        k = jnp.dot(x, attn_params['wk']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        v = jnp.dot(x, attn_params['wv']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(head_dim)
        scores = scores + mask[:seq_len, :seq_len]
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        output = jnp.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return jnp.dot(output, attn_params['wo'])
    
    def transformer_block(x, block_params, n_head, mask, cos, sin):
        h = rms_norm(x, block_params['norm1'])
        x = x + self_attention(h, block_params['attn'], n_head, mask, cos, sin)
        h = rms_norm(x, block_params['norm2'])
        x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, block_params['mlp']['w1'])), block_params['mlp']['w2'])
        return x
    
    def forward(params, tokens, n_head, mask, cos, sin):
        x = params['token_emb'][tokens].astype(jnp.bfloat16)
        for block in params['blocks']:
            x = transformer_block(x, block, n_head, mask, cos, sin)
        x = rms_norm(x, params['final_norm'])
        return jnp.dot(x, params['output_head']).astype(jnp.float32)
    
    # Precompute
    def create_mask(seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -jnp.inf)
    
    def precompute_rope(seq_len, head_dim, theta=10000.0):
        dims = jnp.arange(0, head_dim, 2)
        freqs = 1.0 / (theta ** (dims / head_dim))
        pos = jnp.arange(seq_len)
        angles = jnp.outer(pos, freqs)
        angles = jnp.repeat(angles, 2, axis=-1)
        return jnp.cos(angles), jnp.sin(angles)
    
    mask = create_mask(block_size)
    head_dim = n_embd // n_head
    cos, sin = precompute_rope(block_size, head_dim)
    
    # Optimizer - lower LR for fine-tuning
    learning_rate = 2e-5
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=learning_rate,
        warmup_steps=500,
        decay_steps=steps,
        end_value=1e-6
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, tokens, targets):
        def loss_fn(p):
            logits = forward(p, tokens, n_head, mask, cos, sin)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(target_log_probs)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Training loop
    print(f"\nFine-tuning for {steps} steps...")
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for step in range(steps):
        max_start = len(train_data) - block_size - 1
        starts = rng.integers(0, max_start, size=(batch_size,))
        tokens = np.stack([train_data[i:i + block_size] for i in starts])
        targets = np.stack([train_data[i + 1:i + 1 + block_size] for i in starts])
        
        params, opt_state, loss = train_step(params, opt_state, jnp.array(tokens), jnp.array(targets))
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * batch_size * block_size / max(elapsed, 1)
            print(f"Step {step:5d} | Loss {float(loss):.4f} | {tok_per_sec:.0f} tok/s")
        
        if step > 0 and step % 5000 == 0:
            ckpt = {
                'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
                'step': step,
                'config': cfg
            }
            with open(f"/checkpoints/nanochat_large_ft_step_{step}.pkl", 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"  Saved checkpoint")
            volume.commit()
    
    # Final save
    ckpt = {
        'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
        'step': steps,
        'config': cfg
    }
    with open(f"/checkpoints/nanochat_large_final.pkl", 'wb') as f:
        pickle.dump(ckpt, f)
    volume.commit()
    
    print(f"\nFine-tuning complete! Final loss: {float(loss):.4f}")
    print("Download with: modal volume get nanochat-checkpoints nanochat_large_final.pkl")
    
    return float(loss)


# ============================================================================
# INSTRUCTION FINE-TUNING (The "Chatbot" Upgrade)
# ============================================================================
model_volume = modal.Volume.from_name("nanochat-model", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,
    volumes={"/model": model_volume},
)
def finetune_instruct(steps: int = 5000):
    """Fine-tune the existing nanochat_large_final.pkl on Alpaca instructions."""
    import os
    import pickle
    import time
    import json
    import urllib.request
    
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    import tiktoken
    
    print("=" * 60)
    print("Instruction Fine-tuning NanoChat")
    print("=" * 60)
    
    # Load pretrained model from the model volume
    ckpt_path = "/model/nanochat_large_final.pkl"
    print(f"Loading pretrained model from {ckpt_path}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    params = jax.tree.map(jnp.array, ckpt['params'])
    cfg = ckpt['config']
    print(f"  Config: {cfg}")
    
    # Load Alpaca dataset
    print("\nLoading Alpaca dataset...")
    alpaca_url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    alpaca_path = "/tmp/alpaca.json"
    urllib.request.urlretrieve(alpaca_url, alpaca_path)
    
    with open(alpaca_path, 'r') as f:
        alpaca_data = json.load(f)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def format_example(ex):
        instruction = ex.get('instruction', '')
        inp = ex.get('input', '')
        output = ex.get('output', '')
        prompt = f"User: {instruction}"
        if inp:
            prompt += f" {inp}"
        prompt += f"\nAssistant: {output}\n"
        return prompt
    
    all_tokens = []
    for ex in alpaca_data:
        tokens = enc.encode(format_example(ex))
        all_tokens.extend(tokens)
    
    train_data = np.array(all_tokens, dtype=np.int32)
    print(f"  Training tokens: {len(train_data):,}")
    
    block_size = cfg['block_size']
    n_head = cfg['n_head']
    n_embd = cfg['n_embd']
    n_layer = cfg['n_layer']
    batch_size = 8
    
    def rms_norm(x, gamma, eps=1e-6):
        ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
    
    def apply_rope(xq, xk, cos, sin):
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)
        cos = cos[None, None, :, :].astype(xq.dtype)
        sin = sin[None, None, :, :].astype(xq.dtype)
        return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)
    
    def self_attention(x, attn_params, n_head, mask, cos, sin):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // n_head
        q = jnp.dot(x, attn_params['wq']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        k = jnp.dot(x, attn_params['wk']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        v = jnp.dot(x, attn_params['wv']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(head_dim)
        scores = scores + mask[:seq_len, :seq_len]
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        output = jnp.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return jnp.dot(output, attn_params['wo'])
    
    def transformer_block(x, block_params, n_head, mask, cos, sin):
        h = rms_norm(x, block_params['norm1'])
        x = x + self_attention(h, block_params['attn'], n_head, mask, cos, sin)
        h = rms_norm(x, block_params['norm2'])
        x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, block_params['mlp']['w1'])), block_params['mlp']['w2'])
        return x
    
    def forward(params, tokens, n_head, mask, cos, sin):
        x = params['token_emb'][tokens].astype(jnp.bfloat16)
        for block in params['blocks']:
            x = transformer_block(x, block, n_head, mask, cos, sin)
        x = rms_norm(x, params['final_norm'])
        return jnp.dot(x, params['output_head']).astype(jnp.float32)
    
    def create_mask(seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -jnp.inf)
    
    def precompute_rope(seq_len, head_dim, theta=10000.0):
        dims = jnp.arange(0, head_dim, 2)
        freqs = 1.0 / (theta ** (dims / head_dim))
        pos = jnp.arange(seq_len)
        angles = jnp.outer(pos, freqs)
        angles = jnp.repeat(angles, 2, axis=-1)
        return jnp.cos(angles), jnp.sin(angles)
    
    mask = create_mask(block_size)
    head_dim = n_embd // n_head
    cos, sin = precompute_rope(block_size, head_dim)
    
    learning_rate = 1e-5
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.1)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, tokens, targets):
        def loss_fn(p):
            logits = forward(p, tokens, n_head, mask, cos, sin)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(target_log_probs)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    print(f"\nFine-tuning for {steps} steps...")
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for step in range(steps):
        max_start = len(train_data) - block_size - 1
        starts = rng.integers(0, max_start, size=(batch_size,))
        tokens = np.stack([train_data[i:i + block_size] for i in starts])
        targets = np.stack([train_data[i + 1:i + 1 + block_size] for i in starts])
        
        params, opt_state, loss = train_step(params, opt_state, jnp.array(tokens), jnp.array(targets))
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * batch_size * block_size / max(elapsed, 1)
            print(f"Step {step:5d} | Loss {float(loss):.4f} | {tok_per_sec:.0f} tok/s")
    
    ckpt = {
        'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
        'step': steps,
        'config': cfg
    }
    with open(f"/model/nanochat_large_instruct.pkl", 'wb') as f:
        pickle.dump(ckpt, f)
    model_volume.commit()
    
    print(f"\nFine-tuning complete! Final loss: {float(loss):.4f}")
    print("Instruct model saved to volume as nanochat_large_instruct.pkl")
    return float(loss)


@app.function(
    image=image,
    gpu="A10G",  # A10G is 3-4x faster than T4
    timeout=14400,  # 4 hours
    volumes={"/checkpoints": volume},
)
def train_alpaca(steps: int = 10000):
    """Train NanoChat on Alpaca dataset using cloud GPU."""
    import os
    import pickle
    import time
    import json
    import urllib.request
    
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    import tiktoken
    
    print("=" * 60)
    print("NanoChat Training on Modal GPU")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    
    # ========== Configuration ==========
    class Config:
        block_size = 512
        vocab_size = 50257  # GPT-2
        n_layer = 6
        n_head = 6
        n_embd = 384
        dropout = 0.0
        dtype = "bfloat16"
        
        batch_size = 16
        learning_rate = 1e-4
        max_steps = steps
        warmup_steps = 500
        weight_decay = 0.1
        grad_clip = 1.0
        
        log_interval = 50
        save_interval = 1000
    
    config = Config()
    
    # ========== Data Loading ==========
    print("\n[1/4] Loading OpenAssistant conversational dataset...")
    
    # Download OpenAssistant conversations (pre-processed version)
    oasst_url = "https://huggingface.co/datasets/timdettmers/openassistant-guanaco/resolve/main/openassistant_best_replies_train.jsonl"
    oasst_path = "/tmp/oasst.jsonl"
    if not os.path.exists(oasst_path):
        urllib.request.urlretrieve(oasst_url, oasst_path)
    
    # Load conversations
    conversations = []
    with open(oasst_path, 'r') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    
    print(f"  Loaded {len(conversations)} conversations")
    
    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    
    SYSTEM_PROMPT = "You are NanoChat, a friendly and helpful AI assistant. You can chat about anything."
    
    def format_conversation(conv):
        """Format a conversation for training."""
        text = conv.get('text', '')
        # The guanaco format has ### Human: and ### Assistant: markers
        # We'll adapt it to our format
        formatted = f"### System: {SYSTEM_PROMPT}\n"
        
        # Parse the conversation
        parts = text.split('### ')
        for part in parts:
            if part.startswith('Human:'):
                formatted += f"### User: {part[6:].strip()}\n"
            elif part.startswith('Assistant:'):
                formatted += f"### NanoChat: {part[10:].strip()}\n"
        
        return formatted + "\n"
    
    all_tokens = []
    for conv in conversations:
        tokens = enc.encode(format_conversation(conv))
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.int32)
    split_idx = int(len(all_tokens) * 0.95)
    train_data = all_tokens[:split_idx]
    val_data = all_tokens[split_idx:]
    
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")
    
    # ========== Model ==========
    print("\n[2/4] Initializing model...")
    
    def rms_norm(x, gamma, eps=1e-6):
        ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
    
    def apply_rope(xq, xk, cos, sin):
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)
        cos = cos[None, None, :, :].astype(xq.dtype)
        sin = sin[None, None, :, :].astype(xq.dtype)
        return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)
    
    def self_attention(x, params, n_head, mask, cos, sin):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // n_head
        
        q = jnp.dot(x, params['wq'])
        k = jnp.dot(x, params['wk'])
        v = jnp.dot(x, params['wv'])
        
        q = q.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        
        q, k = apply_rope(q, k, cos, sin)
        
        scale = jnp.sqrt(jnp.array(head_dim, dtype=x.dtype))
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
        scores = scores + mask.astype(scores.dtype)
        
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        output = jnp.matmul(weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        
        return jnp.dot(output, params['wo'])
    
    def transformer_block(x, params, n_head, mask, cos, sin):
        h = rms_norm(x, params['norm1'])
        x = x + self_attention(h, params['attn'], n_head, mask, cos, sin)
        h = rms_norm(x, params['norm2'])
        x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, params['mlp']['w1'])), params['mlp']['w2'])
        return x
    
    def forward(params, tokens, n_head, mask, cos, sin):
        x = params['token_emb'][tokens].astype(jnp.bfloat16)
        for block_params in params['blocks']:
            x = transformer_block(x, block_params, n_head, mask, cos, sin)
        x = rms_norm(x, params['final_norm'])
        return jnp.dot(x, params['output_head']).astype(jnp.float32)
    
    # Initialize params
    key = jax.random.PRNGKey(42)
    std = 0.02
    
    def init_weight(k, shape, scale=1.0):
        return jax.random.normal(k, shape) * std * scale
    
    k1, k2, k3 = jax.random.split(key, 3)
    residual_scale = 1.0 / np.sqrt(2 * config.n_layer)
    
    params = {
        'token_emb': init_weight(k1, (config.vocab_size, config.n_embd)) * 0.5,
        'blocks': [
            {
                'attn': {
                    'wq': init_weight(jax.random.fold_in(k2, i*4+0), (config.n_embd, config.n_embd)),
                    'wk': init_weight(jax.random.fold_in(k2, i*4+1), (config.n_embd, config.n_embd)),
                    'wv': init_weight(jax.random.fold_in(k2, i*4+2), (config.n_embd, config.n_embd)),
                    'wo': init_weight(jax.random.fold_in(k2, i*4+3), (config.n_embd, config.n_embd), residual_scale),
                },
                'mlp': {
                    'w1': init_weight(jax.random.fold_in(k2, i*2+100), (config.n_embd, config.n_embd * 4)),
                    'w2': init_weight(jax.random.fold_in(k2, i*2+101), (config.n_embd * 4, config.n_embd), residual_scale),
                },
                'norm1': jnp.ones((config.n_embd,)),
                'norm2': jnp.ones((config.n_embd,)),
            }
            for i in range(config.n_layer)
        ],
        'final_norm': jnp.ones((config.n_embd,)),
        'output_head': init_weight(k3, (config.n_embd, config.vocab_size))
    }
    
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {num_params:,}")
    
    # Precompute
    def create_mask(seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -jnp.inf)
    
    def precompute_rope(seq_len, head_dim, theta=10000.0):
        dims = jnp.arange(0, head_dim, 2)
        freqs = 1.0 / (theta ** (dims / head_dim))
        pos = jnp.arange(seq_len)
        angles = jnp.outer(pos, freqs)
        angles = jnp.repeat(angles, 2, axis=-1)
        return jnp.cos(angles), jnp.sin(angles)
    
    mask = create_mask(config.block_size)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rope(config.block_size, head_dim)
    
    # ========== Training ==========
    print("\n[3/4] Starting training...")
    
    # Optimizer
    warmup_fn = optax.linear_schedule(0.0, config.learning_rate, config.warmup_steps)
    decay_fn = optax.cosine_decay_schedule(config.learning_rate, config.max_steps - config.warmup_steps)
    schedule = optax.join_schedules([warmup_fn, decay_fn], [config.warmup_steps])
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, tokens, targets):
        def loss_fn(p):
            logits = forward(p, tokens, config.n_head, mask, cos, sin)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(target_log_probs)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Training loop
    rng = np.random.default_rng(42)
    start_time = time.time()
    
    for step in range(config.max_steps):
        # Get batch
        max_start = len(train_data) - config.block_size - 1
        starts = rng.integers(0, max_start, size=(config.batch_size,))
        tokens = np.stack([train_data[i:i + config.block_size] for i in starts])
        targets = np.stack([train_data[i + 1:i + 1 + config.block_size] for i in starts])
        
        params, opt_state, loss = train_step(params, opt_state, jnp.array(tokens), jnp.array(targets))
        
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * config.batch_size * config.block_size / max(elapsed, 1)
            print(f"Step {step:5d} | Loss {float(loss):.4f} | {tok_per_sec:.0f} tok/s")
        
        if step > 0 and step % config.save_interval == 0:
            ckpt = {
                'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
                'step': step,
                'config': {
                    'block_size': config.block_size,
                    'vocab_size': config.vocab_size,
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_embd': config.n_embd,
                }
            }
            with open(f"/checkpoints/checkpoint_step_{step}.pkl", 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"  Saved checkpoint")
            volume.commit()
    
    # Final save
    print("\n[4/4] Saving final checkpoint...")
    ckpt = {
        'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
        'step': config.max_steps,
        'config': {
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
        }
    }
    with open(f"/checkpoints/nanochat_alpaca_final.pkl", 'wb') as f:
        pickle.dump(ckpt, f)
    volume.commit()
    
    print(f"\nTraining complete! Total time: {(time.time() - start_time)/60:.1f} min")
    print("Download checkpoint with: modal volume get nanochat-checkpoints nanochat_alpaca_final.pkl")
    
    return float(loss)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={"/checkpoints": volume},
)
def finetune_conversation(steps: int = 20000):
    """Fine-tune the Alpaca model on conversational data."""
    import os
    import pickle
    import time
    import json
    import urllib.request
    
    import jax
    import jax.numpy as jnp
    import optax
    import numpy as np
    import tiktoken
    
    print("=" * 60)
    print("NanoChat - Fine-tuning on Conversational Data")
    print("=" * 60)
    
    # Load the Alpaca-trained checkpoint
    ckpt_path = "/checkpoints/nanochat_alpaca_final.pkl"
    print(f"Loading base model from {ckpt_path}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    params = jax.tree.map(jnp.array, ckpt['params'])
    cfg = ckpt['config']
    print(f"  Config: {cfg}")
    
    # Load Dolly-15k dataset (100% English, high quality)
    print("\nLoading Dolly-15k dataset...")
    dolly_url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    dolly_path = "/tmp/dolly.jsonl"
    if not os.path.exists(dolly_path):
        urllib.request.urlretrieve(dolly_url, dolly_path)
    
    examples = []
    with open(dolly_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"  Loaded {len(examples)} examples")
    
    enc = tiktoken.get_encoding("gpt2")
    SYSTEM_PROMPT = "You are NanoChat, a friendly and helpful AI assistant. You can chat about anything."
    
    def format_dolly(ex):
        """Format Dolly example for training."""
        instruction = ex.get('instruction', '').strip()
        context = ex.get('context', '').strip()
        response = ex.get('response', '').strip()
        
        formatted = f"### System: {SYSTEM_PROMPT}\n"
        
        # Add context if present
        if context:
            formatted += f"### User: {instruction}\n\nContext: {context}\n"
        else:
            formatted += f"### User: {instruction}\n"
        
        formatted += f"### NanoChat: {response}\n\n"
        return formatted
    
    all_tokens = []
    for ex in examples:
        tokens = enc.encode(format_dolly(ex))
        all_tokens.extend(tokens)
    
    train_data = np.array(all_tokens[:int(len(all_tokens) * 0.95)], dtype=np.int32)
    print(f"  Training tokens: {len(train_data):,}")
    
    # Config
    block_size = cfg['block_size']
    n_head = cfg['n_head']
    batch_size = 16
    learning_rate = 1e-4
    
    # Precompute
    def create_mask(seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -jnp.inf)
    
    def precompute_rope(seq_len, head_dim, theta=10000.0):
        dims = jnp.arange(0, head_dim, 2)
        freqs = 1.0 / (theta ** (dims / head_dim))
        pos = jnp.arange(seq_len)
        angles = jnp.outer(pos, freqs)
        angles = jnp.repeat(angles, 2, axis=-1)
        return jnp.cos(angles), jnp.sin(angles)
    
    mask = create_mask(block_size)
    head_dim = cfg['n_embd'] // n_head
    cos, sin = precompute_rope(block_size, head_dim)
    
    # Model functions (same as before)
    def rms_norm(x, gamma, eps=1e-6):
        ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
    
    def apply_rope(xq, xk, cos, sin):
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)
        cos = cos[None, None, :, :].astype(xq.dtype)
        sin = sin[None, None, :, :].astype(xq.dtype)
        return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)
    
    def self_attention(x, params, n_head, mask, cos, sin):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // n_head
        q = jnp.dot(x, params['wq'])
        k = jnp.dot(x, params['wk'])
        v = jnp.dot(x, params['wv'])
        q = q.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos, sin)
        scale = jnp.sqrt(jnp.array(head_dim, dtype=x.dtype))
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
        scores = scores + mask.astype(scores.dtype)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        output = jnp.matmul(weights, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return jnp.dot(output, params['wo'])
    
    def transformer_block(x, params, n_head, mask, cos, sin):
        h = rms_norm(x, params['norm1'])
        x = x + self_attention(h, params['attn'], n_head, mask, cos, sin)
        h = rms_norm(x, params['norm2'])
        x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, params['mlp']['w1'])), params['mlp']['w2'])
        return x
    
    def forward(params, tokens, n_head, mask, cos, sin):
        x = params['token_emb'][tokens].astype(jnp.bfloat16)
        for block_params in params['blocks']:
            x = transformer_block(x, block_params, n_head, mask, cos, sin)
        x = rms_norm(x, params['final_norm'])
        return jnp.dot(x, params['output_head']).astype(jnp.float32)
    
    # Optimizer - lower learning rate for fine-tuning
    learning_rate = 5e-5  # Lower LR to avoid destroying pretrained weights
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=200,
        decay_steps=steps,
        end_value=1e-6
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.1)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, tokens, targets):
        def loss_fn(p):
            logits = forward(p, tokens, n_head, mask, cos, sin)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(target_log_probs)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Training loop
    print(f"\nFine-tuning for {steps} steps...")
    rng = np.random.default_rng(123)
    start_time = time.time()
    
    for step in range(steps):
        max_start = len(train_data) - block_size - 1
        starts = rng.integers(0, max_start, size=(batch_size,))
        tokens = np.stack([train_data[i:i + block_size] for i in starts])
        targets = np.stack([train_data[i + 1:i + 1 + block_size] for i in starts])
        
        params, opt_state, loss = train_step(params, opt_state, jnp.array(tokens), jnp.array(targets))
        
        if step % 50 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) * batch_size * block_size / max(elapsed, 1)
            print(f"Step {step:5d} | Loss {float(loss):.4f} | {tok_per_sec:.0f} tok/s")
        
        if step > 0 and step % 2000 == 0:
            ckpt = {
                'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
                'step': step,
                'config': cfg
            }
            with open(f"/checkpoints/nanochat_chat_step_{step}.pkl", 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"  Saved checkpoint")
            volume.commit()
    
    # Final save
    ckpt = {
        'params': jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params),
        'step': steps,
        'config': cfg
    }
    with open(f"/checkpoints/nanochat_chat_final.pkl", 'wb') as f:
        pickle.dump(ckpt, f)
    volume.commit()
    
    print(f"\nFine-tuning complete! Final loss: {float(loss):.4f}")
    print("Download with: modal volume get nanochat-checkpoints nanochat_chat_final.pkl")
    return float(loss)


@app.function(
    image=image,
    gpu="T4",  # Cheaper GPU for inference
    timeout=300,
    volumes={"/checkpoints": volume},
)
def chat_gpu(message: str, temperature: float = 0.7):
    """Run inference on GPU - much faster than local CPU."""
    import pickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tiktoken
    
    # Load model - prefer large fine-tuned model
    ckpt_paths = [
        "/checkpoints/nanochat_large_final.pkl",
        "/checkpoints/nanochat_large_pretrained.pkl", 
        "/checkpoints/nanochat_chat_final.pkl",
    ]
    
    ckpt_path = None
    for path in ckpt_paths:
        import os
        if os.path.exists(path):
            ckpt_path = path
            break
    
    if not ckpt_path:
        return "Error: No checkpoint found"
    
    print(f"Loading {ckpt_path}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    params = jax.tree.map(jnp.array, ckpt['params'])
    cfg = ckpt['config']
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Format prompt (casual style - model was trained on this)
    prompt = f"User: {message}\nAssistant:"
    tokens = enc.encode(prompt)
    
    # Model functions
    def rms_norm(x, gamma, eps=1e-6):
        ms = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) * jax.lax.rsqrt(ms + eps) * gamma).astype(x.dtype)
    
    def apply_rope(xq, xk, cos, sin):
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)
        cos = cos[None, None, :, :].astype(xq.dtype)
        sin = sin[None, None, :, :].astype(xq.dtype)
        return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)
    
    def self_attention(x, attn_params, n_head, mask, cos, sin):
        batch, seq_len, n_embd = x.shape
        head_dim = n_embd // n_head
        q = jnp.dot(x, attn_params['wq']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        k = jnp.dot(x, attn_params['wk']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        v = jnp.dot(x, attn_params['wv']).reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(head_dim)
        scores = scores + mask[:seq_len, :seq_len]
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        output = jnp.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return jnp.dot(output, attn_params['wo'])
    
    def forward(params, tokens, n_head, mask, cos, sin):
        x = params['token_emb'][tokens].astype(jnp.bfloat16)
        for block in params['blocks']:
            h = rms_norm(x, block['norm1'])
            x = x + self_attention(h, block['attn'], n_head, mask, cos, sin)
            h = rms_norm(x, block['norm2'])
            x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, block['mlp']['w1'])), block['mlp']['w2'])
        x = rms_norm(x, params['final_norm'])
        return jnp.dot(x, params['output_head']).astype(jnp.float32)
    
    # Precompute
    block_size = cfg['block_size']
    n_head = cfg['n_head']
    head_dim = cfg['n_embd'] // n_head
    mask = jnp.where(jnp.tril(jnp.ones((block_size, block_size))), 0.0, -jnp.inf)
    dims = jnp.arange(0, head_dim, 2)
    freqs = 1.0 / (10000.0 ** (dims / head_dim))
    pos = jnp.arange(block_size)
    angles = jnp.outer(pos, freqs)
    angles = jnp.repeat(angles, 2, axis=-1)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    
    forward_jit = jax.jit(lambda t: forward(params, t, n_head, mask, cos, sin))
    
    # Generate
    key = jax.random.PRNGKey(42)
    for _ in range(100):  # max tokens
        logits = forward_jit(jnp.array(tokens)[None])
        logits = logits[0, -1] / temperature
        probs = jax.nn.softmax(logits)
        key, subkey = jax.random.split(key)
        next_token = jax.random.categorical(subkey, jnp.log(probs))
        tokens.append(int(next_token))
        
        # Stop conditions
        decoded = enc.decode(tokens)
        if "### User:" in decoded or "### System:" in decoded:
            break
    
    # Extract response
    full_text = enc.decode(tokens)
    if "### NanoChat:" in full_text:
        response = full_text.split("### NanoChat:")[-1]
    else:
        response = full_text
    for marker in ["### User:", "### System:"]:
        if marker in response:
            response = response.split(marker)[0]
    
    return response.strip()


@app.local_entrypoint()
def main(steps: int = 10000, resume_from: int = 0, finetune: bool = False, chat: str = "", large: bool = False, finetune_large_model: bool = False):
    """Run training or chat on Modal.
    
    Args:
        steps: Number of training steps
        resume_from: Step number to resume from (0 = start fresh)
        finetune: If True, fine-tune existing Alpaca model on conversational data
        chat: If provided, run inference with this message (uses GPU)
        large: If True, train the 85M parameter model on FineWeb-Edu
        finetune_large_model: If True, fine-tune the large pretrained model
    """
    if chat:
        print("Running inference on GPU...")
        response = chat_gpu.remote(chat)
        print(f"\nNanoChat: {response}")
        return
    
    if finetune_large_model:
        print("=" * 60)
        print("Fine-tuning LARGE model on diverse instructions")
        print("Training on 3 formats: formal, casual, simple Q&A")
        print("This teaches the model to handle real user inputs")
        print("=" * 60)
        final_loss = finetune_large.remote(steps=steps)
        print(f"Fine-tuning finished with final loss: {final_loss:.4f}")
        print("\nTo download: modal volume get nanochat-checkpoints nanochat_large_final.pkl")
        return
    
    if large:
        print("=" * 60)
        print("Training LARGE model (85M params) on FineWeb-Edu")
        print("This will take 6-10 hours and cost ~$20-30")
        print("=" * 60)
        final_loss = train_large.remote(steps=steps)
        print(f"Training finished with final loss: {final_loss:.4f}")
        print("\nTo download: modal volume get nanochat-checkpoints nanochat_large_pretrained.pkl")
        return
    
    if finetune:
        print("Fine-tuning NanoChat on conversational data...")
        final_loss = finetune_conversation.remote(steps=steps)
    elif resume_from > 0:
        print(f"Resuming NanoChat training from step {resume_from}...")
        final_loss = finetune_conversation.remote(steps=steps)
    else:
        print("Starting NanoChat training from scratch...")
        final_loss = train_alpaca.remote(steps=steps)
    
    print(f"Training finished with final loss: {final_loss:.4f}")
    print("\nTo download the checkpoint:")
    print("  modal volume get nanochat-checkpoints nanochat_alpaca_final.pkl ./checkpoints/")
