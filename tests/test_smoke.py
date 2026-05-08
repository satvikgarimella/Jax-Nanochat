import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp

import data_loader
from generate import generate, generate_greedy, top_k_filter
from init import init_params
from train import TrainConfig, create_optimizer
from utils import create_causal_mask, precompute_rope


@dataclass(frozen=True)
class TinyConfig:
    block_size: int = 8
    vocab_size: int = 16
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 8
    dropout: float = 0.0
    dtype: str = "float32"


class NanoChatSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = TinyConfig()
        self.params = init_params(jax.random.PRNGKey(0), self.config)
        head_dim = self.config.n_embd // self.config.n_head
        self.mask = create_causal_mask(self.config.block_size)
        self.cos, self.sin = precompute_rope(self.config.block_size, head_dim)

    def test_generate_keeps_running_when_prompt_fills_context(self):
        prompt = list(range(self.config.block_size))

        generated = generate(
            self.params,
            prompt,
            self.config,
            self.mask,
            self.cos,
            self.sin,
            max_new_tokens=2,
            key=jax.random.PRNGKey(1),
        )
        greedy = generate_greedy(
            self.params,
            prompt,
            self.config,
            self.mask,
            self.cos,
            self.sin,
            max_new_tokens=2,
        )

        self.assertEqual(len(generated), len(prompt) + 2)
        self.assertEqual(len(greedy), len(prompt) + 2)

    def test_top_k_filter_clamps_to_vocab_size(self):
        logits = jnp.arange(self.config.vocab_size, dtype=jnp.float32)
        filtered = top_k_filter(logits, 99)
        self.assertEqual(filtered.shape, logits.shape)

    def test_download_shakespeare_uses_existing_legacy_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "input.txt"
            legacy_path.write_text("To be, or not to be", encoding="utf-8")

            with mock.patch.object(data_loader, "DATA_DIR", tmpdir):
                self.assertEqual(data_loader.download_shakespeare(), "To be, or not to be")

    def test_create_optimizer_handles_short_smoke_runs(self):
        config = TrainConfig(max_steps=1, warmup_steps=100)
        optimizer = create_optimizer(config)
        params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
        state = optimizer.init(params)
        self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
