"""Local chess playground: PYTHONPATH=src uv run python playground.py."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from taktiny import nn

from artoria import Artoria, ArtoriaConfig
from artoria.tokenizer import ArtoriaTokenizer
from artoria.moves import move_to_id


def game_state(fen: str, moves: list[str]):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError('FEN must describe a valid standard chess position.')
    history, notation = [board.fen(en_passant='fen')], []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f'Illegal move: {uci}')
        notation.append(board.san(move))
        board.push(move)
        history.append(board.fen(en_passant='fen'))
    return board, history or [board.fen(en_passant='fen')], notation


def choose_move(board, history, score_moves):
    """Select the highest-scoring legal action, with immediate mate priority."""
    if board.outcome(claim_draw=True) is not None:
        return None
    candidates = list(board.legal_moves)
    for move in candidates:
        child = board.copy(stack=True)
        child.push(move)
        # Always take an immediate checkmate, regardless of model calibration.
        if child.is_checkmate():
            return move
    scores = score_moves(board, history, candidates)
    if len(scores) != len(candidates) or any(not math.isfinite(x) for x in scores):
        raise ValueError('Model returned invalid move scores.')
    # Policy logits and board likelihoods represent the side to move's action,
    # so BOTH colors maximize them; only the eval head is White-perspective.
    return candidates[max(range(len(candidates)), key=lambda i: scores[i])]


class Evaluator:
    def __init__(self, checkpoint: Path, move_head: str = 'policy'):
        model = Artoria(ArtoriaConfig(), rngs=nn.Rngs(0))
        leaves, structure = jax.tree.flatten(model)
        target = {str(i): value for i, value in enumerate(leaves)}
        with ocp.StandardCheckpointer() as checkpointer:
            restored = checkpointer.restore(str(checkpoint.resolve() / 'model_state'), target=target)
        if set(restored) != set(target):
            raise ValueError('Checkpoint does not match the current model configuration.')
        for key, value in restored.items():
            if value.shape != target[key].shape:
                raise ValueError(f'Checkpoint shape mismatch at leaf {key}.')
        self.model = jax.tree.unflatten(structure, [restored[str(i)] for i in range(len(leaves))])
        self.tokenizer = ArtoriaTokenizer()
        # Fixed input length avoids recompilation on every move; select the last
        # real position. Causal attention prevents padding from affecting it.
        self.forward = jax.jit(lambda model, ids, last: tuple(output[0, last] for output in model(ids)))
        self.move_head = move_head
        state = json.loads((checkpoint / 'trainer_state' / 'metadata').read_text())
        self.info = {'checkpoint': checkpoint.name, 'step': state['global_step'],
                     'parameters': sum(math.prod(x.shape) for x in leaves), 'move_head': move_head}

    def predict(self, history):
        history = history[-128:]
        ids = self.tokenizer.encode(history)
        ids = np.pad(ids, ((0, 0), (0, 128 - len(history)), (0, 0)), mode='edge')
        return self.forward(self.model, jnp.asarray(ids), jnp.asarray(len(history) - 1))

    def evaluate(self, history):
        _, evaluation, _ = self.predict(history)
        score = float(evaluation[0])
        if not math.isfinite(score):
            raise ValueError('Model returned a non-finite evaluation.')
        return score, min(len(history), 128)

    def score_moves(self, board, history, candidates):
        board_logits, _, policy_logits = self.predict(history)
        if self.move_head == 'policy':
            logits = np.asarray(policy_logits, dtype=np.float32)
            return [float(logits[move_to_id(move)]) for move in candidates]
        log_probs = np.asarray(jax.nn.log_softmax(board_logits.astype(jnp.float32), axis=-1))
        scores = []
        for move in candidates:
            child = board.copy()
            child.push(move)
            tokens = self.tokenizer.encode(child.fen(en_passant='fen'))[0, 0]
            scores.append(float(log_probs[np.arange(70), tokens].sum()))
        return scores


def make_handler(evaluator):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/', '/index.html'):
                self.send_error(404)
                return
            data = (Path(__file__).parent / 'playground' / 'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path not in ('/api/evaluate', '/api/play'):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 65536:
                    raise ValueError('Request is too large or empty.')
                body = json.loads(self.rfile.read(length))
                moves = body.get('moves', [])
                if not isinstance(moves, list) or len(moves) > 2048 or any(not isinstance(m, str) for m in moves):
                    raise ValueError('Moves must be a list of UCI strings (maximum 2048).')
                board, history, notation = game_state(body.get('fen', chess.STARTING_FEN), moves)
                moves = moves.copy()
                reply = None
                if self.path == '/api/play':
                    player = body.get('player', 'white')
                    if player not in ('white', 'black'):
                        raise ValueError('Player must be white or black.')
                    if board.turn != (player == 'white'):
                        move = choose_move(board, history, evaluator.score_moves)
                        if move is not None:
                            reply = move.uci()
                            moves.append(reply)
                            board, history, notation = game_state(body.get('fen', chess.STARTING_FEN), moves)
                score, context = evaluator.evaluate(history)
                outcome = board.outcome(claim_draw=True)
                result = {**evaluator.info, 'fen': board.fen(en_passant='fen'),
                          'pieces': {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
                          'legal_moves': [m.uci() for m in board.legal_moves], 'san': notation,
                          'turn': 'White' if board.turn else 'Black', 'check': board.is_check(),
                          'result': outcome.result() if outcome else None,
                          'termination': outcome.termination.name.replace('_', ' ').lower() if outcome else None,
                          'score': score, 'context': context, 'moves': moves, 'reply': reply}
                status = 200
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                status, result = 400, {'error': str(exc)}
            except Exception:
                import traceback
                traceback.print_exc()
                status, result = 500, {'error': 'Evaluation failed. See the server terminal.'}
            data = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('ckpt/artoria-01'))
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--move-head', choices=('policy', 'board'), default='policy',
                        help='Rank legal moves by the policy head or next-board likelihood.')
    args = parser.parse_args()
    print(f'Loading {args.checkpoint} and warming up inference…', flush=True)
    evaluator = Evaluator(args.checkpoint, args.move_head)
    evaluator.evaluate([chess.STARTING_FEN])
    server = HTTPServer(('127.0.0.1', args.port), make_handler(evaluator))
    print(f'Playground ready: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
