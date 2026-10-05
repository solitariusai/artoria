import unittest
import chess
from artoria.moves import MOVE_UCIS, NUM_MOVES, move_to_id


class MoveVocabularyTests(unittest.TestCase):
    def test_unique_round_trip_and_special_moves(self):
        self.assertEqual(NUM_MOVES, 4208)
        self.assertEqual(len(set(MOVE_UCIS)), NUM_MOVES)
        for move in ('e2e4', 'e1g1', 'e1c1', 'e8g8', 'e5d6', 'a7a8q', 'b2a1n', 'h7g8r'):
            self.assertEqual(MOVE_UCIS[move_to_id(chess.Move.from_uci(move))], move)
        with self.assertRaises(KeyError):
            move_to_id('0000')

    def test_every_promotion_is_distinct(self):
        ids = {move_to_id('a7a8' + p) for p in 'qrbn'}
        self.assertEqual(len(ids), 4)
