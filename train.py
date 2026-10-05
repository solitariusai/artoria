import argparse
import io
import logging
import os
import re
from typing import Any, NotRequired, TypedDict

import chess.pgn
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from jax.sharding import AxisType
from taktiny import nn
from taktiny.data import DataLoader, FlatMap, Pack, train_validation_split
from taktiny.trainer import DatasetConfig, Trainer, TrainingConfig
from taktiny.utils import map_logical_axis_names

from artoria import Artoria, ArtoriaConfig
from artoria.cache import ArtoriaCache  # noqa: F401
from artoria.tokenizer import ArtoriaTokenizer
from artoria.moves import move_to_id


class PositionEval(TypedDict):
    fen: str
    eval: float | str | None
    next_move: NotRequired[str | None]


_EVAL_PATTERN = re.compile(
    r"\[%eval\s+(?P<score>\#[+-]?\d+|[+-]?\d+(?:\.\d+)?)(?:\s*,\s*\d+)?\s*\]"
)

_LOGGER = logging.getLogger(__name__)


class _StrictGameBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error: Exception) -> None:
        raise ValueError(f"Invalid PGN: {error}") from error

def moves_to_fen(data: str, train: bool = False, *, policy: bool = False) -> list[PositionEval]:
    """Convert annotated PGN into positions after each mainline move.

    Accepts bare movetext (as in this file) or complete PGN games, including FEN
    starting-position headers. Multiple games are processed in input order.
    Evaluations remain from White's perspective: numeric scores are in pawns,
    mate scores are strings such as '#3', and missing evaluations are None.
    Clocks, prose comments, and variations are ignored. Invalid games raise
    ValueError rather than returning a partially parsed training sequence.
    With train=True, fen retains piece placement, side to move, castling rights,
    and en passant target; only the halfmove and fullmove counters are removed.
    With policy=True, include each game's starting board and attach its next
    UCI move to every position. Final positions have next_move=None.
    """
    positions: list[PositionEval] = []
    source = io.StringIO(data)
    while (game := chess.pgn.read_game(source, Visitor=_StrictGameBuilder)) is not None:
        if game.errors:
            raise ValueError(f"Invalid PGN: {game.errors[0]}") from game.errors[0]
        board = game.board()
        nodes = list(game.mainline())
        if policy and nodes:
            fen = board.fen(en_passant='fen')
            positions.append({'fen': ' '.join(fen.split()[:4]) if train else fen,
                              'eval': None, 'next_move': nodes[0].move.uci()})
        for index, node in enumerate(nodes):
            board.push(node.move)
            match = _EVAL_PATTERN.search(node.comment)
            score = match.group("score") if match else None
            evaluation = (
                float(score) if score is not None and not score.startswith("#") else score
            )
            fen = board.fen(en_passant="fen")
            if train:
                fen = " ".join(fen.split()[:4])
            position: PositionEval = {"fen": fen, "eval": evaluation}
            if policy:
                position['next_move'] = nodes[index + 1].move.uci() if index + 1 < len(nodes) else None
            positions.append(position)
    return positions


def tokenize_game(row: dict[str, Any], *, warn: bool = True) -> list[dict[str, Any]]:
    """Emit encoded games, or skip a malformed/unsupported record with a warning.

    Include starting boards and aligned next-move labels. Only finite pawn
    evaluations contribute to regression; missing/mate scores have eval_mask=0.
    """
    identity = row.get("GameURL") or row.get("Site") or "unknown game"
    variant = row.get("Variant")
    if variant and variant.lower() not in ("standard", "chess", "normal"):
        if warn:
            _LOGGER.warning("Skipping %s: unsupported variant %s", identity, variant)
        return []
    movetext = row["movetext"]
    if row.get("FEN"):
        # Restore starting-position metadata when stored separately from movetext.
        starting_fen = row["FEN"]
        movetext = f'[SetUp "1"]\n[FEN "{starting_fen}"]\n\n{movetext}'
    try:
        positions = moves_to_fen(movetext, train=True, policy=True)
        if not positions:
            return []
    except ValueError as error:
        if warn:
            _LOGGER.warning("Skipping %s: %s; movetext begins %r", identity, error, row["movetext"][:160])
        return []
    games, start = [], 0
    for end, position in enumerate(positions):
        if position['next_move'] is not None:
            continue
        game = positions[start:end + 1]
        start = end + 1
        scores = np.zeros(len(game), dtype=np.float32)
        score_mask = np.zeros(len(game), dtype=np.int32)
        policy_ids = np.zeros(len(game), dtype=np.int32)
        policy_mask = np.zeros(len(game), dtype=np.int32)
        for index, pos in enumerate(game):
            score = pos['eval']
            if isinstance(score, (int, float)) and np.isfinite(score):
                scores[index], score_mask[index] = score, 1
            if pos['next_move'] is not None:
                policy_ids[index], policy_mask[index] = move_to_id(pos['next_move']), 1
        tokens = ArtoriaTokenizer().encode([p['fen'] for p in game])[0]
        games.append({'fen_ids': tokens, 'eval': scores, 'eval_mask': score_mask,
                      'policy_ids': policy_ids, 'policy_mask': policy_mask})
    if len(games) > 64:
        raise ValueError('A dataset record may contain at most 64 games.')
    return games


class _GameValidator(chess.pgn.BaseVisitor[bool]):
    """Check SAN legality without building a game tree, FENs, or token arrays."""

    def begin_game(self) -> None:
        self.mainline_moves = 0
        self.variation_depth = 0

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        if self.variation_depth == 0:
            self.mainline_moves += 1

    def visit_board(self, board: chess.Board) -> None:
        if self.variation_depth == 0:
            if any(right not in "KQkq-" for right in board.castling_xfen()):
                raise ValueError("Castling rights cannot be encoded by this tokenizer.")
            if board.ep_square is not None:
                expected_rank = 5 if board.turn == chess.WHITE else 2
                if chess.square_rank(board.ep_square) != expected_rank:
                    raise ValueError("En passant target has an invalid rank.")

    def begin_variation(self) -> None:
        self.variation_depth += 1

    def end_variation(self) -> None:
        self.variation_depth -= 1

    def result(self) -> bool:
        return self.mainline_moves > 0


def is_trainable_game(row: dict[str, Any]) -> bool:
    """Validate supported PGN quietly without tokenizing every position."""
    variant = row.get("Variant")
    if variant and variant.lower() not in ("standard", "chess", "normal"):
        return False
    movetext = row["movetext"]
    if row.get("FEN"):
        movetext = f'[SetUp "1"]\n[FEN "{row["FEN"]}"]\n\n{movetext}'
    source = io.StringIO(movetext)
    has_moves = False
    try:
        while (valid := chess.pgn.read_game(source, Visitor=_GameValidator)) is not None:
            has_moves = has_moves or valid
    except ValueError:
        return False
    return has_moves


def filter_training_games(dataset, workers: int = 1):
    """Remove unsupported, invalid, and empty games before splitting/loading.

    Hugging Face caches the selected indices for datasets backed by cache files,
    so subsequent runs can reuse the filtering result.
    """
    if workers < 1:
        raise ValueError("Filter workers must be positive.")
    filtered = dataset.filter(
        is_trainable_game,
        num_proc=min(workers, len(dataset)) if workers > 1 and len(dataset) > 1 else None,
        desc="Filtering valid chess games",
    )
    print(f"Dataset filter: kept {len(filtered):,} of {len(dataset):,} games "
          f"(removed {len(dataset) - len(filtered):,}).")
    if len(filtered) < 2:
        raise ValueError("Need at least two valid games for a train/validation split.")
    return filtered


def training_loss(model, batch):
    fen_ids = batch["fen_ids"]
    position_ids = batch["position_ids"]
    segment_ids = jnp.cumsum(position_ids == 0, axis=-1) - 1
    same_game = segment_ids[..., :, None] == segment_ids[..., None, :]
    mask = jnp.tril(same_game)[:, None, ...]
    logits, ev_logits, policy_logits = model(fen_ids, mask, position_ids)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :-1].astype(jnp.float32), fen_ids[:, 1:]
    )
    # Do not learn a next-position target across packed game boundaries.
    next_mask = (position_ids[:, 1:] == position_ids[:, :-1] + 1)[..., None]
    token_count = jnp.maximum(next_mask.sum() * fen_ids.shape[-1], 1)
    token_loss = jnp.where(next_mask, token_loss, 0).sum() / token_count
    predictions = ev_logits.squeeze(-1)
    eval_loss = optax.huber_loss(predictions, batch["eval"], delta=0.3)
    eval_mask = batch["eval_mask"].astype(bool)
    eval_loss = jnp.where(eval_mask, eval_loss, 0).sum() / jnp.maximum(eval_mask.sum(), 1)
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(
        policy_logits.astype(jnp.float32), batch['policy_ids'])
    policy_mask = batch['policy_mask'].astype(bool)
    policy_loss = jnp.where(policy_mask, policy_loss, 0).sum() / jnp.maximum(policy_mask.sum(), 1)
    return policy_loss + token_loss + 0.1 * eval_loss

def process_dataset(repo: str, max_len: int, batch_size: int, workers: int, filter_workers: int = 1):
    ds = load_dataset(repo, split='train')
    ds = filter_training_games(ds, workers=filter_workers)
    train, val = train_validation_split(ds, 0.01)

    train_loader = DataLoader(
        train, 
        operations=[
            FlatMap(tokenize_game, max_fan_out=64),
            Pack(max_len, keys=('fen_ids', 'eval', 'eval_mask', 'policy_ids', 'policy_mask'), position_key='position_ids', drop_remainder=True)
        ],
        worker_buffer_size=2,
        worker_count=workers,
        read_options=grain.ReadOptions(
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        axis_names={
            'fen_ids': ('batch', 'sequence', 'board'),
            'eval': ('batch', 'sequence'),
            'eval_mask': ('batch', 'sequence'),
            'policy_ids': ('batch', 'sequence'),
            'policy_mask': ('batch', 'sequence'),
            'position_ids': ('batch', 'sequence'),
        },
        batch_size=batch_size,
        drop_remainder=True,
    )
    val_loader = DataLoader(
        val, 
        operations=[
            FlatMap(tokenize_game, max_fan_out=64),
            Pack(max_len, keys=('fen_ids', 'eval', 'eval_mask', 'policy_ids', 'policy_mask'), position_key='position_ids', drop_remainder=True)
        ],
        worker_buffer_size=2,
        worker_count=workers,
        read_options=grain.ReadOptions(
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        axis_names={
            'fen_ids': ('batch', 'sequence', 'board'),
            'eval': ('batch', 'sequence'),
            'eval_mask': ('batch', 'sequence'),
            'policy_ids': ('batch', 'sequence'),
            'policy_mask': ('batch', 'sequence'),
            'position_ids': ('batch', 'sequence'),
        },
        batch_size=batch_size,
        drop_remainder=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data-repo', type=str, default='Lichess/tournament-chess-games')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--filter-workers', type=int, default=min(8, os.cpu_count() or 1))

    # train
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--loss-chunk-size', type=int, default=128)
    
    # other
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--not-save', action='store_false', default=True) # debug

    args = parser.parse_args()
    if args.loss_chunk_size < 1:
        parser.error('--loss-chunk-size must be positive')
    if args.filter_workers < 1:
        parser.error('--filter-workers must be positive')

    # Finish CPU preprocessing before initializing the JAX backend and its threads.
    train_loader, val_loader = process_dataset(
        args.data_repo,
        args.max_seq_len,
        args.batch_size,
        args.workers,
        filter_workers=args.filter_workers,
    )

    mesh = jax.make_mesh(
        (1, jax.device_count()), 
        ('model', 'data'), 
        (AxisType.Auto, AxisType.Auto)
    )
    jax.set_mesh(mesh)
    map_logical_axis_names({
        'vocab': None,
        'hidden': 'model',
        'num_heads': None,
        'head_dim': None,
        'intermediate': None,
        'batch': 'data',
        'sequence': None,
    })

    config = ArtoriaConfig()
    schedule = optax.cosine_decay_schedule(args.lr, args.max_steps)
    optimizer = optax.adamw(schedule, weight_decay=args.wd)

    model = Artoria(config, rngs=nn.Rngs(0))
    trainer = Trainer(
        model,
        TrainingConfig(
            max_steps=args.max_steps,
            schedule=schedule,
            optimizer=optimizer,
            eval_strategy='steps' if args.eval else 'no',
            eval_steps=args.max_steps // 4 if args.max_steps > 10 else args.max_steps,
            output_dir=f'{args.out_dir}',
            save_at_end=args.not_save,
            log_interval=args.log_interval,
        ),
        DatasetConfig(
            train_loader,
            val_loader
        ),
        loss_fn=training_loss,
    )

    trainer.train()
