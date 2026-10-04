import unittest

import numpy as np

from artoria import ArtoriaConfig, ArtoriaTokenizer


class ArtoriaTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ArtoriaTokenizer()
        self.start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
        self.d4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3"

    def test_piece_ids_square_order_and_turn(self):
        actual = self.tokenizer.encode(self.start)
        expected = (
            [9, 7, 8, 10, 11, 8, 7, 9] + [6] * 8 + [12] * 32
            + [0] * 8 + [3, 1, 2, 4, 5, 2, 1, 3] + [14, 16, 16, 16, 16, 17]
        )
        self.assertEqual(actual.shape, (1, 1, 70))
        self.assertEqual(actual.dtype, np.int32)
        np.testing.assert_array_equal(actual[0, 0], expected)
        self.assertEqual(self.tokenizer.vocab_size, ArtoriaConfig().vocab)

    def test_batch_sequence_and_empty_square_runs(self):
        actual = self.tokenizer.encode([[self.start, self.d4], [self.d4, self.start]])
        self.assertEqual(actual.shape, (2, 2, 70))
        np.testing.assert_array_equal(actual[0, 1, 32:40], [12, 12, 12, 0, 12, 12, 12, 12])
        np.testing.assert_array_equal(actual[:, :, 64], [[14, 13], [13, 14]])
        np.testing.assert_array_equal(actual[0, 1, 48:56], [0, 0, 0, 12, 0, 0, 0, 0])

    def test_single_game_and_full_fen(self):
        short = self.tokenizer.encode([self.start, self.d4])
        full = self.tokenizer.encode([self.start + " 0 1", self.d4 + " 0 1"])
        self.assertEqual(short.shape, (1, 2, 70))
        np.testing.assert_array_equal(short, full)

    def test_return_list_with_unequal_game_lengths(self):
        actual = self.tokenizer.encode([[self.start], [self.start, self.d4]], return_list=True)
        self.assertIsInstance(actual, list)
        self.assertEqual([len(game) for game in actual], [1, 2])
        self.assertEqual(actual[0][0], self.tokenizer.encode(self.start)[0, 0].tolist())
        self.assertEqual(actual[1][1], self.tokenizer.encode(self.d4)[0, 0].tolist())
        self.assertTrue(all(len(position) == 70 for game in actual for position in game))
        self.assertIsInstance(actual[1][1][0], int)

    def test_return_list_preserves_axes_and_empty_games(self):
        self.assertEqual(self.tokenizer.encode(self.start, return_list=True),
                         self.tokenizer.encode(self.start).tolist())
        self.assertEqual(self.tokenizer.encode([self.start, self.d4], return_list=True),
                         self.tokenizer.encode([self.start, self.d4]).tolist())
        self.assertEqual(self.tokenizer.encode([[]], return_list=True), [[]])

    def test_invalid_inputs(self):
        for invalid in ([], [[]], [[self.start], [self.start, self.d4]],
                        "8/8/8/8/8/8/8 w - -", "8/8/8/8/8/8/8/7 w - -",
                        "8/8/8/8/8/8/8/9 w - -", "8/8/8/8/8/8/8/X7 w - -",
                        "8/8/8/8/8/8/8/8 x - -", "8/8/8/8/8/8/8/8"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.tokenizer.encode(invalid)

    def test_castling_flags_distinguish_identical_boards(self):
        board = "r3k2r/8/8/8/8/8/8/R3K2R"
        for rights, expected in (("KQkq", [16, 16, 16, 16]),
                                 ("Kq", [16, 15, 15, 16]),
                                 ("Qk", [15, 16, 16, 15]),
                                 ("-", [15, 15, 15, 15])):
            with self.subTest(rights=rights):
                actual = self.tokenizer.encode(f"{board} w {rights} -")[0, 0]
                np.testing.assert_array_equal(actual[65:69], expected)
                self.assertEqual(actual[69], 17)

    def test_en_passant_files_for_both_turns(self):
        for turn, rank in (("w", "6"), ("b", "3")):
            for index, file in enumerate("abcdefgh"):
                with self.subTest(turn=turn, file=file):
                    actual = self.tokenizer.encode(f"8/8/8/8/8/8/8/8 {turn} - {file}{rank}")
                    self.assertEqual(actual[0, 0, 69], 18 + index)

    def test_invalid_special_move_fields(self):
        for suffix in ("w KK -", "w K- -", "w A -", "w - d3", "b - d6", "w - i6", "w - a", "w"):
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):
                    self.tokenizer.encode(f"8/8/8/8/8/8/8/8 {suffix}")


if __name__ == "__main__":
    unittest.main()
