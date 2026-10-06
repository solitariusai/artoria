from dataclasses import dataclass
from artoria.moves import NUM_MOVES


@dataclass
class ArtoriaConfig:
    vocab: int = 26 # black + white + space + turn + castling + en passant
    n_moves: int = NUM_MOVES
    d_model: int = 384
    d_latent: int = 1280
    n_parallel: int = 6
    n_depth: int = 4
    n_spatial_layers: int = 4
    spatial_chunk_size: int = 64  # Target boards/group; minimum one timestep per batch.
    epsilon: float = 1e-7
    n_heads: int = 8
    n_kv_heads: int = 8
    head_dim: int = 32
    dtype: str = 'bfloat16'


__all__ = ['ArtoriaConfig']
