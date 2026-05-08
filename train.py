"""
NanoChat Training Script
Supports both Shakespeare (character-level) and Alpaca (instruction fine-tuning).
"""

import os
import pickle
import time
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass, replace

from model import forward, cast_to_dtype
from init import init_params
from utils import create_causal_mask, precompute_rope
from data_loader import (
    prepare_shakespeare, prepare_alpaca, 
    DataLoader, get_batch_for_eval
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model architecture
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    vocab_size: int = None
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 5000
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 250
    save_interval: int = 500
    eval_batches: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # Precision
    dtype: str = "float32"  # "float32" or "bfloat16"
    
    # Dataset
    dataset: str = "shakespeare"  # "shakespeare" or "alpaca"


# Preset configurations
SHAKESPEARE_CONFIG = TrainConfig(
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=128,
    batch_size=32,
    max_steps=5000,
    dtype="float32",
    dataset="shakespeare",
)

ALPACA_CONFIG = TrainConfig(
    block_size=512,
    n_layer=6,
    n_head=6,
    n_embd=384,
    batch_size=8,
    max_steps=20000,
    learning_rate=1e-4,
    warmup_steps=500,
    dtype="bfloat16",
    dataset="alpaca",
)


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(params, step, config, checkpoint_dir="checkpoints"):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Convert to float32 for saving
    params_f32 = jax.tree.map(lambda x: np.array(x.astype(jnp.float32)), params)
    
    checkpoint = {
        'params': params_f32,
        'step': step,
        'config': {
            'block_size': config.block_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'vocab_size': config.vocab_size,
            'dtype': config.dtype,
            'dataset': config.dataset,
        }
    }
    
    path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  Saved -> {path}")
    return path


def load_checkpoint(path):
    """Load model checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    params = jax.tree.map(jnp.array, checkpoint['params'])
    return params, checkpoint['step'], checkpoint['config']


# ============================================================================
# Training Functions
# ============================================================================

def create_optimizer(config):
    """Create optimizer with optional warmup and weight decay."""
    # Learning rate schedule with warmup
    if config.warmup_steps > 0:
        if config.max_steps <= config.warmup_steps:
            schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=config.learning_rate,
                transition_steps=max(config.max_steps, 1),
            )
        else:
            warmup_fn = optax.linear_schedule(
                init_value=0.0,
                end_value=config.learning_rate,
                transition_steps=config.warmup_steps
            )
            decay_fn = optax.cosine_decay_schedule(
                init_value=config.learning_rate,
                decay_steps=config.max_steps - config.warmup_steps
            )
            schedule = optax.join_schedules(
                schedules=[warmup_fn, decay_fn],
                boundaries=[config.warmup_steps]
            )
    else:
        schedule = config.learning_rate
    
    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    return optimizer


def create_train_state(config, key):
    """Initialize model and optimizer."""
    params = init_params(key, config)
    
    # Cast to training dtype if using bfloat16
    if config.dtype == "bfloat16":
        params = cast_to_dtype(params, jnp.bfloat16)
    
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(params)
    
    return params, optimizer, opt_state


def make_train_step(optimizer, config, mask, cos, sin):
    """Create JIT-compiled training step."""
    
    @jax.jit
    def train_step(params, opt_state, tokens, targets):
        def loss_fn(p):
            logits = forward(p, tokens, config, mask, cos, sin)
            # Cross-entropy loss
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
            return -jnp.mean(target_log_probs)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    return train_step


def make_eval_step(config, mask, cos, sin):
    """Create JIT-compiled evaluation step."""
    
    @jax.jit
    def eval_step(params, tokens, targets):
        logits = forward(params, tokens, config, mask, cos, sin)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
        return -jnp.mean(target_log_probs)
    
    return eval_step


def estimate_loss(params, eval_fn, data, config, num_batches=10):
    """Estimate loss on dataset."""
    rng = np.random.default_rng(seed=1337)
    losses = []
    for _ in range(num_batches):
        tokens, targets = get_batch_for_eval(data, config.block_size, config.batch_size, rng)
        losses.append(float(eval_fn(params, tokens, targets)))
    return np.mean(losses)


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config: TrainConfig = None, resume_from: str = None):
    """Main training function."""
    
    if config is None:
        config = replace(SHAKESPEARE_CONFIG)
    
    print("=" * 60)
    print(f"NanoChat Training - {config.dataset.upper()} Mode")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    if config.dataset == "alpaca":
        try:
            train_data, val_data, tokenizer = prepare_alpaca(val_split=0.05)
            config.vocab_size = tokenizer.vocab_size
        except ImportError as e:
            print(f"Error: {e}")
            print("Falling back to Shakespeare dataset...")
            config.dataset = "shakespeare"
            train_data, val_data, tokenizer = prepare_shakespeare(val_split=0.1)
            config.vocab_size = tokenizer.vocab_size
    else:
        train_data, val_data, tokenizer = prepare_shakespeare(val_split=0.1)
        config.vocab_size = tokenizer.vocab_size
    
    train_loader = DataLoader(train_data, config.block_size, config.batch_size)
    
    # Initialize or resume model
    print("\n[2/4] Initializing model...")
    key = jax.random.PRNGKey(42)
    start_step = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"  Resuming from {resume_from}")
        params, start_step, saved_config = load_checkpoint(resume_from)
        optimizer = create_optimizer(config)
        opt_state = optimizer.init(params)
    else:
        params, optimizer, opt_state = create_train_state(config, key)
    
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Architecture: {config.n_layer}L-{config.n_head}H-{config.n_embd}D")
    print(f"  Parameters: {num_params:,}")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Precision: {config.dtype}")
    
    # Precompute
    mask = create_causal_mask(config.block_size)
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rope(config.block_size, head_dim)
    
    # Create training functions
    train_step = make_train_step(optimizer, config, mask, cos, sin)
    eval_step = make_eval_step(config, mask, cos, sin)
    
    # Training loop
    print(f"\n[3/4] Training for {config.max_steps} steps...")
    print("-" * 60)
    
    start_time = time.time()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for step in range(start_step, config.max_steps):
        # Get batch
        tokens, targets = train_loader.get_batch()
        
        # Training step
        params, opt_state, loss = train_step(params, opt_state, tokens, targets)
        train_losses.append(float(loss))
        
        # Logging
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = np.mean(train_losses[-config.log_interval:]) if train_losses else float(loss)
            tok_per_sec = (step - start_step + 1) * config.batch_size * config.block_size / max(elapsed, 1)
            print(f"Step {step:5d} | Loss {avg_loss:.4f} | {tok_per_sec:.0f} tok/s")
        
        # Validation
        if step > 0 and step % config.eval_interval == 0:
            train_loss = estimate_loss(params, eval_step, train_data, config, config.eval_batches)
            val_loss = estimate_loss(params, eval_step, val_data, config, config.eval_batches)
            val_losses.append((step, val_loss))
            print(f"  >> Eval: train={train_loss:.4f} | val={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(params, step, config, config.checkpoint_dir)
        
        # Periodic checkpoint
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(params, step, config, config.checkpoint_dir)
    
    # Final checkpoint
    save_checkpoint(params, config.max_steps, config, config.checkpoint_dir)
    
    # Summary
    print("-" * 60)
    print(f"\n[4/4] Training complete!")
    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Save history
    with open(os.path.join(config.checkpoint_dir, 'history.pkl'), 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses, 'config': config}, f)
    
    return params, config, tokenizer


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NanoChat")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "alpaca"],
                        help="Dataset to train on")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # Select config preset
    if args.dataset == "alpaca":
        config = replace(ALPACA_CONFIG)
    else:
        config = replace(SHAKESPEARE_CONFIG)
    
    # Override with CLI args
    if args.steps:
        config.max_steps = args.steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
