from dataclasses import dataclass


@dataclass
class ArtoriaConfig:
    vocab: int = 26 # black + white + space + turn + castling + en passant
    d_model: int = 128
    d_latent: int = 512
    n_parallel: int = 6
    n_depth: int = 4
    n_spatial_layers: int = 1
    epsilon: float = 1e-7
    n_heads: int = 12
    n_kv_heads: int = 12
    head_dim: int = 64
    dtype: str = 'bfloat16'


__all__ = ['ArtoriaConfig']
