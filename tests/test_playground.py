import unittest

import chess
import jax.numpy as jnp

from playground import Evaluator, choose_move, game_state
from artoria.moves import move_to_id, NUM_MOVES
from artoria.tokenizer import ArtoriaTokenizer


class PlaygroundTests(unittest.TestCase):
    def test_history_tracks_positions_after_moves(self):
        board, history, san = game_state(chess.STARTING_FEN, ['e2e4', 'e7e5', 'g1f3'])
        self.assertEqual(san, ['e4', 'e5', 'Nf3'])
        self.assertEqual(len(history), 4)
        self.assertEqual(history[-1], board.fen(en_passant='fen'))
        self.assertEqual(history[1].split()[3], 'e3')


    def test_rejects_illegal_moves_and_invalid_positions(self):
        with self.assertRaisesRegex(ValueError, 'Illegal move'):
            game_state(chess.STARTING_FEN, ['e2e5'])
        with self.assertRaisesRegex(ValueError, 'valid standard'):
            game_state('8/8/8/8/8/8/8/8 w - - 0 1', [])


    def test_special_moves(self):
        board, _, san = game_state('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', ['e1g1'])
        self.assertEqual(san, ['O-O'])
        self.assertEqual(board.piece_at(chess.F1).piece_type, chess.ROOK)
        board, _, _ = game_state('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', ['e5d6'])
        self.assertIsNone(board.piece_at(chess.D5))
        board, _, _ = game_state('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', ['a7a8n'])
        self.assertEqual(board.piece_at(chess.A8).piece_type, chess.KNIGHT)

    def test_policy_selection_maximizes_for_both_colors_and_preserves_board(self):
        for moves, preferred in [([], 'e2e4'), (['e2e4'], 'e7e5')]:
            board, history, _ = game_state(chess.STARTING_FEN, moves)
            original = board.fen()
            def evaluate(position, context, candidates):
                self.assertEqual(position.fen(), original)
                self.assertEqual(context, history)
                return [2.0 if move.uci() == preferred else 0.0 for move in candidates]

            selected = choose_move(board, history, evaluate)
            self.assertEqual(selected.uci(), preferred)
            self.assertEqual(board.fen(), original)

    def test_immediate_mate_and_finished_game(self):
        board, history, _ = game_state(chess.STARTING_FEN, ['f2f3', 'e7e5', 'g2g4'])

        def unexpected_evaluation(_):
            self.fail('Terminal positions should use chess rules.')

        move = choose_move(board, history, unexpected_evaluation)
        self.assertEqual(move.uci(), 'd8h4')
        board.push(move)
        self.assertIsNone(choose_move(board, history, unexpected_evaluation))
        draw = chess.Board('4k3/8/8/8/8/8/8/4K3 w - - 0 1')
        self.assertIsNone(choose_move(draw, [], unexpected_evaluation))

    def test_playground_policy_masks_illegal_actions(self):
        engine = Evaluator.__new__(Evaluator)
        engine.move_head = 'policy'
        policy = jnp.zeros(NUM_MOVES).at[move_to_id('e2e5')].set(1000)
        policy = policy.at[move_to_id('e2e4')].set(10)
        engine.predict = lambda history: (None, jnp.array([-1000.]), policy)
        board, history, _ = game_state(chess.STARTING_FEN, [])
        self.assertEqual(choose_move(board, history, engine.score_moves).uci(), 'e2e4')

    def test_board_likelihood_uses_next_board_head_for_both_colors(self):
        engine = Evaluator.__new__(Evaluator)
        engine.move_head = 'board'
        engine.tokenizer = ArtoriaTokenizer()
        for moves, preferred in [([], 'e2e4'), (['e2e4'], 'e7e5')]:
            board, history, _ = game_state(chess.STARTING_FEN, moves)
            child = board.copy()
            child.push_uci(preferred)
            tokens = engine.tokenizer.encode(child.fen(en_passant='fen'))[0, 0]
            logits = jnp.zeros((70, 26)).at[jnp.arange(70), tokens].set(10)
            engine.predict = lambda history: (logits, jnp.array([1000.]), None)
            self.assertEqual(choose_move(board, history, engine.score_moves).uci(), preferred)
