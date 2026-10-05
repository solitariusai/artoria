import unittest

import jax
import jax.numpy as jnp
import numpy as np
from taktiny import nn

from artoria import Artoria, ArtoriaConfig
from artoria.impl import ArtoriaAttention
from artoria.rope import ArtoriaRoPE
from taktiny.utils.trainer import _combine_params, _parameter_labels, _partition_params
from train import training_loss


class AttentionShapeTests(unittest.TestCase):
    def setUp(self):
        self.config = ArtoriaConfig(
            d_model=16, d_latent=32, n_heads=4, n_kv_heads=2,
            head_dim=8, n_depth=1, n_parallel=1, dtype="float32",
        )

    def test_grouped_query_rope_keeps_q_and_k_four_dimensional(self):
        attention = ArtoriaAttention(self.config, rngs=nn.Rngs(0))
        rope = ArtoriaRoPE(self.config.head_dim)
        x = jnp.ones((140, 3, 16))
        positions = jnp.array([[0, 1, 2], [0, 0, 1]])
        output = attention(x, position_embedding=rope(positions))
        self.assertEqual(output.shape, x.shape)

    def test_batched_outputs_match_individual_games_with_distinct_masks(self):
        model = Artoria(self.config, rngs=nn.Rngs(0))
        ids = jnp.arange(2 * 4 * 70).reshape(2, 4, 70) % 26
        positions = jnp.array([[0, 1, 2, 3], [0, 1, 0, 1]])
        segments = jnp.cumsum(positions == 0, axis=-1)
        mask = jnp.tril(segments[:, :, None] == segments[:, None, :])[:, None]
        batched = jax.jit(lambda i, m, p: model(i, m, p))(ids, mask, positions)
        self.assertEqual(batched[0].shape, (2, 4, 70, 26))
        self.assertEqual(batched[1].shape, (2, 4, 70, 1))
        for game in range(2):
            single = model(ids[game:game + 1], mask[game:game + 1], positions[game:game + 1])
            for actual, expected in zip(batched, single):
                np.testing.assert_allclose(actual[game:game + 1], expected, rtol=2e-5, atol=2e-5)

    def test_future_and_other_packed_games_cannot_change_past_outputs(self):
        model = Artoria(self.config, rngs=nn.Rngs(1))
        ids = jnp.arange(2 * 4 * 70).reshape(2, 4, 70) % 26
        positions = jnp.array([[0, 1, 2, 3], [0, 1, 0, 1]])
        segments = jnp.cumsum(positions == 0, axis=-1)
        mask = jnp.tril(segments[:, :, None] == segments[:, None, :])[:, None]
        original = model(ids, mask, positions)[0]
        future_changed = model(ids.at[:, 3].set(12), mask, positions)[0]
        np.testing.assert_allclose(original[:, :3], future_changed[:, :3], rtol=2e-5, atol=2e-5)
        earlier_game_changed = model(ids.at[1, :2].set(12), mask, positions)[0]
        np.testing.assert_allclose(original[1, 2:], earlier_game_changed[1, 2:], rtol=2e-5, atol=2e-5)

    def test_checkpointed_nested_stacks_support_training_gradients(self):
        self.config.n_depth = 2
        self.config.n_parallel = 2
        model = Artoria(self.config, rngs=nn.Rngs(2))
        trainable, frozen = _partition_params(model, _parameter_labels(model))
        batch = {
            "fen_ids": jnp.arange(2 * 3 * 70).reshape(2, 3, 70) % 26,
            "position_ids": jnp.array([[0, 1, 2], [0, 1, 0]]),
            "eval": jnp.array([[0.2, 0.1, 0.0], [-0.1, 0.2, 0.3]]),
            "eval_mask": jnp.ones((2, 3), dtype=jnp.int32),
        }
        loss, gradients = jax.jit(jax.value_and_grad(
            lambda parameters: training_loss(_combine_params(parameters, frozen), batch)
        ))(trainable)
        self.assertTrue(np.isfinite(float(loss)))
        for leaf in jax.tree.leaves(gradients):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        self.assertGreater(float(jnp.linalg.norm(gradients.ranks.value)), 0)
        self.assertGreater(float(jnp.linalg.norm(gradients.files.value)), 0)


if __name__ == "__main__":
    unittest.main()
