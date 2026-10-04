import jax
import jax.numpy as jnp
from taktiny import nn


class ArtoriaRoPE(nn.Module):
    """Rotary cosines and sines for the split-half attention rotation."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        if head_dim <= 0 or head_dim % 2:
            raise ValueError('RoPE head dimension must be positive and even')
        if base <= 0:
            raise ValueError('RoPE base must be positive')

        self.frequencies = nn.Parameter(
            base ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim),
            trainable=False,
        )

    def __call__(
        self, 
        positions: jax.Array, 
        *, 
        dtype: jnp.dtype = jnp.float32
    ) -> tuple[jax.Array, jax.Array]:
        """Return (cos, sin), each shaped [..., sequence, head_dim]."""
        if positions.ndim == 1:
            positions = positions[None]

        freq = self.frequencies.value
        angles = positions.astype(jnp.float32)[..., None] * freq[None, ...]
        angles = jnp.concatenate((angles, angles), axis=-1)
        return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


__all__ = ['ArtoriaRoPE']