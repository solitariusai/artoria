import jax
import jax.numpy as jnp
from taktiny import nn

from artoria.conf import ArtoriaConfig


class ArtoriaCache(nn.Pytree):
    def __init__(
        self, 
        config: ArtoriaConfig, 
        num_batches: int, 
        max_sequences: int = 256, 
        dtype: str | None = None, 
        num_layers: int | None = None
    ):
        num_key_value_heads = config.n_kv_heads
        head_dims = config.head_dim
        num_layers = num_layers if num_layers is not None else config.n_depth * config.n_parallel
        if dtype is None:
            self.dtype = config.dtype

        self.k_cache = jax.new_ref(jnp.zeros((num_batches, num_layers, max_sequences, num_key_value_heads, head_dims), dtype=self.dtype))
        self.v_cache = jax.new_ref(jnp.zeros((num_batches, num_layers, max_sequences, num_key_value_heads, head_dims), dtype=self.dtype))
        self.position_idx = jax.new_ref(jnp.asarray(0, dtype=jnp.int32))
        self.cache_length = max_sequences

    def get(self, layer_idx: int | jax.Array):
        return self.k_cache[:, layer_idx], self.v_cache[:, layer_idx]

    def update(self, key: jax.Array, value: jax.Array, layer_idx: jax.Array | int):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError('key and value must have matching [batch, sequence, heads, dim] shapes')
        if key.shape[0] != self.k_cache.shape[0] or key.shape[2:] != self.k_cache.shape[3:]:
            raise ValueError('key and value shapes do not match the cache')
        if key.shape[1] > self.k_cache.shape[2]:
            raise ValueError('sequence is longer than the cache capacity')

        length = key.shape[1]
        idx = self.position_idx[...]
        self.k_cache[:, layer_idx, jax.ds(idx, length)] = key.astype(self.dtype)
        self.v_cache[:, layer_idx, jax.ds(idx, length)] = value.astype(self.dtype)
        
    def advance(self, length):
        self.position_idx[...] += length