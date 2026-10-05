import unittest
from unittest.mock import patch
import io
from contextlib import redirect_stdout

import jax
import jax.numpy as jnp
import numpy as np
from taktiny.data import DataLoader, FlatMap, Pack
from datasets import Dataset

from train import filter_training_games, is_trainable_game, process_dataset, tokenize_game, training_loss
from artoria.moves import move_to_id, NUM_MOVES


class TrainingPipelineTests(unittest.TestCase):
    def test_fast_validator_matches_full_conversion(self):
        rows = [
            {"movetext": "1. e4 { [%eval 0.2] } e5 (1... c5) 2. Nf3 *"},
            {"movetext": "1. e4 (1. Bh6) e5 *"},  # Illegal variation also rejects the game.
            {"movetext": "1. e4 e5 2. Bh6 *"},
            {"movetext": "*"},
            {"movetext": ""},
            {"movetext": "1. d4 *\n\n1. e4 e5 2. Bh6 *"},
            {"movetext": "1. e4 *", "Variant": "Chess960"},
            {"movetext": "1. a8=Q+ *", "FEN": "7k/P7/8/8/8/8/8/7K w - - 0 1"},
            {"movetext": "1. e4 *", "FEN": "invalid"},
        ]
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(is_trainable_game(row), bool(tokenize_game(row, warn=False)))
        with patch("train.ArtoriaTokenizer", side_effect=AssertionError("Must not tokenize")):
            self.assertTrue(is_trainable_game(rows[0]))

    def test_filter_removes_bad_games_before_loading_without_warnings(self):
        dataset = Dataset.from_list([
            {"movetext": "1. e4 { [%eval 0.2] } e5 *", "Variant": "Standard"},
            {"movetext": "1. d4 e5 2. dxe5 Nc6 3. Nd3 *", "Variant": "Standard"},
            {"movetext": "1. e4 *", "Variant": "Chess960"},
            {"movetext": "1. e4 *", "Variant": "From Position"},
            {"movetext": "", "Variant": "Standard"},
            {"movetext": "1. d4 { [%eval 0.3] } d5 *", "Variant": "Standard"},
        ])
        with patch("train._LOGGER.warning") as warning, redirect_stdout(io.StringIO()) as output:
            filtered = filter_training_games(dataset)
            for row in filtered:
                self.assertTrue(tokenize_game(row))
        warning.assert_not_called()
        self.assertEqual(filtered["movetext"], [dataset[0]["movetext"], dataset[5]["movetext"]])
        self.assertIn("kept 2 of 6 games (removed 4)", output.getvalue())

    def test_all_invalid_dataset_fails_before_split(self):
        dataset = Dataset.from_list([{"movetext": "1. e4 e5 2. Bh6 *"}])
        with patch("train._LOGGER.warning") as warning, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "at least two valid games"):
                filter_training_games(dataset)
        warning.assert_not_called()

    def test_dataset_is_filtered_before_train_validation_split(self):
        dataset = Dataset.from_list([
            {"movetext": "1. e4 *"}, {"movetext": "1. e4 e5 2. Bh6 *"},
            {"movetext": "1. d4 *"},
        ])
        with patch("train.load_dataset", return_value=dataset), \
             patch("train.train_validation_split", side_effect=RuntimeError("reached split")) as split, \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "reached split"):
                process_dataset("example", 128, 1, 0)
        selected = split.call_args.args[0]
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(is_trainable_game(row) for row in selected))

    def test_invalid_game_skipped_whole_and_next_game_preserved(self):
        bad = {"movetext": "1. d4 e5 2. dxe5 Nc6 3. Nd3 *", "GameURL": "bad-game"}
        good = {"movetext": "1. e4 { [%eval 0.25] } e5 { [%eval -0.1] } *"}
        with self.assertLogs("train", level="WARNING") as logs:
            self.assertEqual(tokenize_game(bad), [])
        self.assertIn("bad-game", logs.output[0])
        self.assertIn("Nd3", logs.output[0])
        row = tokenize_game(good)[0]
        self.assertEqual(row["fen_ids"].shape, (3, 70))
        np.testing.assert_allclose(row["eval"], [0, 0.25, -0.1])
        np.testing.assert_array_equal(row['policy_ids'], [move_to_id('e2e4'), move_to_id('e7e5'), 0])
        np.testing.assert_array_equal(row['policy_mask'], [1, 1, 0])

    def test_metadata_and_non_numeric_evaluations(self):
        row = tokenize_game({
            "FEN": "7k/P7/8/8/8/8/8/7K w - - 0 1", "Variant": "Standard",
            "movetext": "1. a8=Q+ { [%eval #3] } Kh7 *",
        })[0]
        self.assertEqual(row["fen_ids"][1, 0], 4)
        self.assertEqual(row["eval"].dtype, np.float32)
        np.testing.assert_array_equal(row["eval_mask"], [0, 0, 0])
        np.testing.assert_array_equal(row["eval"], [0, 0, 0])
        self.assertEqual(row['policy_ids'][0], move_to_id('a7a8q'))
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
            Pack(7, keys=("fen_ids", "eval", "eval_mask", 'policy_ids', 'policy_mask'), position_key="position_ids"),
        ], worker_count=0, batch_size=1)
        with self.assertLogs("train", level="WARNING"):
            batch = next(iter(loader))
        self.assertEqual(batch["fen_ids"].shape, (1, 7, 70))
        np.testing.assert_allclose(batch["eval"], [[0, 0.2, -0.1, 0, 0.3, 0, 0]])
        np.testing.assert_array_equal(batch["eval_mask"], [[0, 1, 1, 0, 1, 0, 0]])
        np.testing.assert_array_equal(batch["position_ids"], [[0, 1, 2, 0, 1, 2, 3]])
        np.testing.assert_array_equal(batch['policy_mask'], [[1, 1, 0, 1, 1, 1, 0]])

    def test_loss_uses_integer_labels_and_masks_game_boundaries(self):
        def model(ids, mask, positions):
            # Only transition 0->1 is a training target; the next game starts at 2.
            logits = jnp.zeros((*ids.shape, 26)).at[:, 1].set(100 * jax.nn.one_hot(ids[:, 1], 26))
            predictions = jnp.ones((*ids.shape[:2], 1))
            return logits, predictions, jnp.zeros((*ids.shape[:2], NUM_MOVES))

        batch = {
            "fen_ids": jnp.zeros((1, 3, 70), dtype=jnp.int32),
            "position_ids": jnp.array([[0, 1, 0]]),
            "eval": jnp.array([[1., 100., -100.]]),
            "eval_mask": jnp.array([[1, 0, 0]]),
            'policy_ids': jnp.zeros((1, 3), dtype=jnp.int32),
            'policy_mask': jnp.array([[1, 0, 0]]),
        }
        actual = jax.jit(lambda b: training_loss(model, b))(batch)
        self.assertAlmostEqual(float(actual), float(np.log(26) + np.log(NUM_MOVES)), places=5)
        batch["position_ids"] = jnp.zeros((1, 3), dtype=jnp.int32)
        batch["eval_mask"] = jnp.zeros((1, 3), dtype=jnp.int32)
        batch['policy_mask'] = jnp.zeros((1, 3), dtype=jnp.int32)
        self.assertEqual(float(training_loss(model, batch)), 0.0)

    def test_multiple_games_keep_separate_policy_sequences(self):
        games = tokenize_game({'movetext': '1. e4 *\n\n1. d4 d5 *'})
        self.assertEqual([len(g['fen_ids']) for g in games], [2, 3])
        self.assertEqual(games[0]['policy_ids'][0], move_to_id('e2e4'))
        self.assertEqual(games[1]['policy_ids'][0], move_to_id('d2d4'))
        self.assertTrue(all(g['policy_mask'][-1] == 0 for g in games))


if __name__ == "__main__":
    unittest.main()
