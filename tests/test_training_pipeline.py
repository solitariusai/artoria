import unittest

import jax
import jax.numpy as jnp
import numpy as np
from taktiny.data import DataLoader, FlatMap, Pack

from train import tokenize_game, training_loss


class TrainingPipelineTests(unittest.TestCase):
    def test_invalid_game_skipped_whole_and_next_game_preserved(self):
        bad = {"movetext": "1. d4 e5 2. dxe5 Nc6 3. Nd3 *", "GameURL": "bad-game"}
        good = {"movetext": "1. e4 { [%eval 0.25] } e5 { [%eval -0.1] } *"}
        with self.assertLogs("train", level="WARNING") as logs:
            self.assertEqual(tokenize_game(bad), [])
        self.assertIn("bad-game", logs.output[0])
        self.assertIn("Nd3", logs.output[0])
        row = tokenize_game(good)[0]
        self.assertEqual(row["fen_ids"].shape, (2, 70))
        np.testing.assert_allclose(row["eval"], [0.25, -0.1])

    def test_metadata_and_non_numeric_evaluations(self):
        row = tokenize_game({
            "FEN": "7k/P7/8/8/8/8/8/7K w - - 0 1", "Variant": "Standard",
            "movetext": "1. a8=Q+ { [%eval #3] } Kh7 *",
        })[0]
        self.assertEqual(row["fen_ids"][0, 0], 4)
        self.assertEqual(row["eval"].dtype, np.float32)
        np.testing.assert_array_equal(row["eval_mask"], [0, 0])
        np.testing.assert_array_equal(row["eval"], [0, 0])
        with self.assertLogs("train", level="WARNING"):
            self.assertEqual(tokenize_game({"Variant": "Atomic", "movetext": "1. e4 *"}), [])

    def test_loader_packs_aligned_fields_and_skips_invalid_records(self):
        rows = [
            {"movetext": "1. e4 { [%eval 0.2] } e5 { [%eval -0.1] } *"},
            {"movetext": "1. e4 e5 2. Bh6 *"},
            {"movetext": "1. d4 { [%eval 0.3] } d5 2. c4 { [%eval #2] } *"},
        ]
        loader = DataLoader(rows, operations=[
            FlatMap(tokenize_game, max_fan_out=1),
            Pack(5, keys=("fen_ids", "eval", "eval_mask"), position_key="position_ids"),
        ], worker_count=0, batch_size=1)
        with self.assertLogs("train", level="WARNING"):
            batch = next(iter(loader))
        self.assertEqual(batch["fen_ids"].shape, (1, 5, 70))
        np.testing.assert_allclose(batch["eval"], [[0.2, -0.1, 0.3, 0, 0]])
        np.testing.assert_array_equal(batch["eval_mask"], [[1, 1, 1, 0, 0]])
        np.testing.assert_array_equal(batch["position_ids"], [[0, 1, 0, 1, 2]])

    def test_loss_uses_integer_labels_and_masks_game_boundaries(self):
        def model(ids, mask, positions):
            # Only transition 0->1 is a training target; the next game starts at 2.
            logits = jnp.zeros((*ids.shape, 26)).at[:, 1].set(100 * jax.nn.one_hot(ids[:, 1], 26))
            predictions = jnp.ones((*ids.shape, 1))
            return logits, predictions

        batch = {
            "fen_ids": jnp.zeros((1, 3, 70), dtype=jnp.int32),
            "position_ids": jnp.array([[0, 1, 0]]),
            "eval": jnp.array([[1., 100., -100.]]),
            "eval_mask": jnp.array([[1, 0, 0]]),
        }
        actual = jax.jit(lambda b: training_loss(model, b))(batch)
        self.assertAlmostEqual(float(actual), float(np.log(26)), places=5)
        batch["position_ids"] = jnp.zeros((1, 3), dtype=jnp.int32)
        batch["eval_mask"] = jnp.zeros((1, 3), dtype=jnp.int32)
        self.assertEqual(float(training_loss(model, batch)), 0.0)


if __name__ == "__main__":
    unittest.main()
