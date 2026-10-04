import unittest

from train import moves_to_fen
from artoria import ArtoriaTokenizer


class MovesToFenTests(unittest.TestCase):
    def test_positions_and_evaluations(self):
        self.assertEqual(
            moves_to_fen("1. d4 { [%eval 0.25] [%clk 1:30:43] } "
                         "1... Nf6 { [%eval -0.22] }"),
            [
                {"fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "eval": 0.25},
                {"fen": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2", "eval": -0.22},
            ],
        )

    def test_annotations_variations_and_missing_eval(self):
        positions = moves_to_fen(
            "1. e4?! { [%eval #3,18] } { Best move was d4. } "
            "(1. d4 { [%eval 0.5] }) e5 { [%clk 0:30:00] } "
            "2. Nf3 { [%eval #-2] } *"
        )
        self.assertEqual([row["eval"] for row in positions], ["#3", None, "#-2"])
        self.assertEqual(positions[-1]["fen"],
                         "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")

    def test_custom_start_and_promotion(self):
        positions = moves_to_fen(
            '[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]\n\n'
            '1. a8=Q+ { [%eval 9.5] } *'
        )
        self.assertEqual(positions, [{"fen": "Q6k/8/8/8/8/8/8/7K b - - 0 1", "eval": 9.5}])

    def test_multiple_games_and_empty_input(self):
        self.assertEqual(moves_to_fen(""), [])
        positions = moves_to_fen("1. e4 { [%eval 0.2] } *\n\n1. d4 { [%eval 0.3] } *")
        self.assertEqual([row["eval"] for row in positions], [0.2, 0.3])
        self.assertIn("3P4", positions[1]["fen"])

    def test_illegal_move_raises(self):
        with self.assertRaises(ValueError):
            moves_to_fen("1. e4 e5 2. Bh6 *")

    def test_training_fen_retains_special_move_state(self):
        moves = "1. e4 { [%eval 0.2] } a6 2. e5 d5 3. exd6 *"
        full = moves_to_fen(moves)
        training = moves_to_fen(moves, train=True)
        self.assertEqual(training, [
            {"fen": " ".join(row["fen"].split()[:4]), "eval": row["eval"]}
            for row in full
        ])
        tokenizer = ArtoriaTokenizer()
        tokens = tokenizer.encode([row["fen"] for row in training])
        self.assertEqual(tokens.shape, (1, 5, 70))
        self.assertEqual(tokens[0, 3, 69], 21)  # d6 after Black's d5
        self.assertEqual(tokens[0, 4, 69], 17)  # target cleared after capture
        self.assertEqual(tokens[0, 4, 19], 0)   # White pawn on d6
        self.assertEqual(tokens[0, 4, 27], 12)  # Captured Black pawn removed from d5

    def test_castling_updates_training_flags(self):
        moves = ('[SetUp "1"]\n[FEN "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"]\n\n'
                 '1. O-O O-O-O *')
        rows = moves_to_fen(moves, train=True)
        tokens = ArtoriaTokenizer().encode([row["fen"] for row in rows])
        self.assertEqual(tokens[0, 0, 65:69].tolist(), [15, 15, 16, 16])
        self.assertEqual(tokens[0, 1, 65:69].tolist(), [15, 15, 15, 15])



if __name__ == "__main__":
    unittest.main()
