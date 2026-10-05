"""Stable standard-chess UCI action vocabulary, including underpromotions."""
import chess


MOVE_UCIS = tuple(
    chess.square_name(source) + chess.square_name(target)
    for source in chess.SQUARES for target in chess.SQUARES if source != target
) + tuple(
    chess.square_name(chess.square(file, rank))
    + chess.square_name(chess.square(target_file, target_rank)) + promotion
    for rank, target_rank in ((6, 7), (1, 0))
    for file in range(8)
    for target_file in range(max(0, file - 1), min(8, file + 2))
    for promotion in 'qrbn'
)
MOVE_TO_ID = {move: index for index, move in enumerate(MOVE_UCIS)}
NUM_MOVES = len(MOVE_UCIS)  # 4208


def move_to_id(move: str | chess.Move) -> int:
    return MOVE_TO_ID[move.uci() if isinstance(move, chess.Move) else move]
