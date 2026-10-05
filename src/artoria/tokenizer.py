from collections.abc import Sequence

import numpy as np


class ArtoriaTokenizer:
    """Encode FEN positions in board order (a8 through h1), then side to move.

    White PNBRQK use IDs 0–5; black pnbrqk use IDs 6–11. Empty squares
    use ID 12; black to move uses 13 and white to move uses 14.
    Four castling flags follow in KQkq order (15 absent, 16 present), then
    the en passant file (17 none, 18–25 for a–h). Accepts four-field training
    FEN or full six-field FEN; move counters are ignored.
    """

    piece_to_id = {piece: index for index, piece in enumerate("PNBRQKpnbrqk")}  # noqa: RUF012
    empty_id = 12
    turn_to_id = {"b": 13, "w": 14}  # noqa: RUF012
    castling_to_id = {False: 15, True: 16}  # noqa: RUF012
    en_passant_to_id = {"-": 17, **{file: 18 + i for i, file in enumerate("abcdefgh")}}  # noqa: RUF012
    vocab_size = 26
    position_length = 70

    def _encode_position(self, fen: str) -> list[int]:
        if not isinstance(fen, str):
            raise TypeError("Each position must be a FEN string.")
        fields = fen.split()
        if len(fields) not in (4, 6):
            raise ValueError("FEN must contain board, turn, castling, and en passant, or all six fields.")
        ranks = fields[0].split("/")
        if len(ranks) != 8:
            raise ValueError("FEN must contain exactly eight ranks.")
        tokens: list[int] = []
        for rank in ranks:
            squares: list[int] = []
            for symbol in rank:
                if symbol in self.piece_to_id:
                    squares.append(self.piece_to_id[symbol])
                elif symbol in "12345678":
                    squares.extend([self.empty_id] * int(symbol))
                else:
                    raise ValueError(f"Invalid FEN board symbol: {symbol!r}.")
            if len(squares) != 8:
                raise ValueError("Each FEN rank must expand to exactly eight squares.")
            tokens.extend(squares)
        if fields[1] not in self.turn_to_id:
            raise ValueError("FEN turn must be 'b' or 'w'.")
        tokens.append(self.turn_to_id[fields[1]])
        castling = fields[2]
        if castling != "-" and (
            any(right not in "KQkq" for right in castling)
            or len(set(castling)) != len(castling)
        ):
            raise ValueError("Castling rights must be '-' or unique letters from KQkq.")
        tokens.extend(self.castling_to_id[right in castling] for right in "KQkq")
        target = fields[3]
        if target == "-":
            tokens.append(self.en_passant_to_id["-"])
        else:
            expected_rank = "6" if fields[1] == "w" else "3"
            if len(target) != 2 or target[0] not in "abcdefgh" or target[1] != expected_rank:
                raise ValueError("En passant target must be '-' or a file on rank 6 for White, 3 for Black.")
            tokens.append(self.en_passant_to_id[target[0]])
        return tokens

    def encode(
        self,
        fens: str | Sequence[str] | Sequence[Sequence[str]],
        return_list: bool = False,
    ) -> np.ndarray | list[list[list[int]]]:
        """Return int32 IDs with shape [B, T, 70].

        A string becomes [1, 1, 70]; a sequence of strings becomes [1, T, 70].
        Nested sequences represent a batch of games.
        Positions retain input order; supply each game's FENs from its start.
        With return_list=True, return nested Python lists [B][T][70], allowing
        unequal game lengths. Otherwise, game lengths must be equal and nonzero.
        No padding token is added.
        """
        if isinstance(fens, str):
            games = [[fens]]
        else:
            entries = list(fens)
            if not entries:
                raise ValueError("At least one position or game is required.")
            if isinstance(entries[0], str):
                games = [entries]
            else:
                if any(isinstance(game, str) for game in entries):
                    raise TypeError("Do not mix FEN strings and game sequences.")
                games = [list(game) for game in entries]
        sequence_length = len(games[0])
        if not return_list and (
            sequence_length == 0 or any(len(game) != sequence_length for game in games)
        ):
            raise ValueError("Games must have the same nonzero number of positions.")
        tokens = [[self._encode_position(fen) for fen in game] for game in games]  # ty: ignore[invalid-argument-type]
        if return_list:
            return tokens
        return np.asarray(tokens, dtype=np.int32)


__all__ = ["ArtoriaTokenizer"]
