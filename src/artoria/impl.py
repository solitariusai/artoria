import jax
import jax.numpy as jnp
from taktiny import nn

from artoria.cache import ArtoriaCache
from artoria.conf import ArtoriaConfig
from artoria.rope import ArtoriaRoPE


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 =  jnp.split(x, 2, axis=-1)
    return jnp.concat([-x2, x1], axis=-1)

class ArtoriaAttention(nn.Module):
    def __init__(self, config: ArtoriaConfig, *, rngs: nn.Rngs):
        self.qkv_proj = nn.Linear(
            config.d_model, 
            (config.n_heads + config.n_kv_heads * 2, config.head_dim), 
            bias=False, 
            dtype=config.dtype, 
            rngs=rngs, 
            axis_names=('embed', 'n_heads', 'head_dim')
        )
        self.o_proj = nn.Linear(
            (config.n_heads, config.head_dim), 
            config.d_model, 
            bias=False, 
            dtype=config.dtype, 
            rngs=rngs, 
            axis_names=('embed', 'n_heads', 'head_dim')
        )
        self.n_kv_heads = config.n_kv_heads

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ArtoriaCache | None = None,
        layer_idx: jax.Array | int | None = None,
    ) -> jax.Array: 
        x = self.qkv_proj(x)
        dtype = x.dtype
        k, v, q = jnp.split(x, [self.n_kv_heads, 2 * self.n_kv_heads], axis=-2)
        if position_embedding is not None:
            cos, sin = position_embedding # shape [T, H]
            if cos.ndim == 3:
                cos = cos[:, :, None, :]
                sin = sin[:, :, None, :]

            elif cos.ndim == 2:
                cos = cos[None, :, None, :]
                sin = sin[None, :, None, :]

            B = cos.shape[0]
            BTq, Sq, Nq, Hq = q.shape
            BTk, Sk, Nk, Hk = k.shape

            q = q.reshape(B, BTq // B, Sq, Nq, Hq)
            k = k.reshape(B, BTk // B, Sk, Nk, Hk)

            q = (q * cos[:, None, ...] + rotate_half(q) * sin[:, None, ...]).astype(dtype)
            k = (k * cos[:, None, ...] + rotate_half(k) * sin[:, None, ...]).astype(dtype)

            q = q.reshape(BTq, Sq, Nq, Hq)
            k = k.reshape(BTk, Sk, Nk, Hk)

        if cache is not None:
            assert layer_idx is not None
            cache.update(k, v, layer_idx)
            k, v= cache.get(layer_idx)
            q_pos = cache.position_idx[...] + jnp.arange(q.shape[1])
            cache_mask = (
                jnp.arange(k.shape[1])[None, :] <= q_pos[:, None]
            )
            mask = cache_mask if mask is None else mask & cache_mask
        
        o = jax.nn.dot_product_attention(
            q, k, v, mask=mask
        )
        return self.o_proj(o).astype(dtype)

class ArtoriaMLP(nn.Module):
    def __init__(self, config: ArtoriaConfig, *, rngs: nn.Rngs):
        self.up_proj = nn.Linear(
            config.d_model, 
            config.d_latent * 2, 
            bias=False, 
            dtype=config.dtype, 
            rngs=rngs, 
            axis_names=('embed', 'latent')
        )
        self.down_proj = nn.Linear(
            config.d_latent, 
            config.d_model, 
            bias=False, 
            dtype=config.dtype, 
            rngs=rngs, 
            axis_names=('latent', 'embed')
        )

    def __call__(self, x: jax.Array):
        x = self.up_proj(x)
        x1, x2 = jnp.split(x, 2, -1)
        return self.down_proj(jax.nn.silu(x1) * x2)

class ArtoriaSpatialEncoder(nn.Module):
    """Mix squares and metadata within each board, independently of time."""

    def __init__(self, config: ArtoriaConfig, *, rngs: nn.Rngs):
        self.attn = ArtoriaAttention(config, rngs=rngs)
        self.mlp = ArtoriaMLP(config, rngs=rngs)
        self.norm1 = nn.RMSNorm(config.d_model, config.epsilon, dtype='float32', axis_names=('embed',))
        self.norm2 = nn.RMSNorm(config.d_model, config.epsilon, dtype='float32', axis_names=('embed',))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.attn(self.norm1(x)) + x
        return self.mlp(self.norm2(x)) + x


class ArtoriaDecoder(nn.Module):
    """Causal attention over one pooled vector per game position."""
    def __init__(self, config: ArtoriaConfig, *, rngs: nn.Rngs):
        self.attn = ArtoriaAttention(config, rngs=rngs)
        self.mlp = ArtoriaMLP(config, rngs=rngs)
        self.norm1 = nn.RMSNorm(config.d_model, config.epsilon, dtype='float32', axis_names=('embed',))
        self.norm2 = nn.RMSNorm(config.d_model, config.epsilon, dtype='float32', axis_names=('embed',))

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ArtoriaCache | None = None,
        layer_idx: jax.Array | int | None = None
    ):
        x = self.attn(self.norm1(x), mask, position_embedding, cache, layer_idx) + x
        return self.mlp(self.norm2(x)) + x

class Artoria(nn.Module):
    """Spatial board encoding -> mean pooling -> temporal game decoding.

    Returns board logits [B,T,70,vocab], evaluations [B,T,1],
    and next-move policy logits [B,T,n_moves].
    """
    def __init__(self, config: ArtoriaConfig, *, rngs: nn.Rngs):
        self.wte = nn.Embedding(config.vocab, config.d_model, dtype=config.dtype, rngs=rngs, axis_names=('vocab', 'embed'))
        self.spatial_layers = nn.List([
            ArtoriaSpatialEncoder(config, rngs=rngs) for _ in range(config.n_spatial_layers)
        ])
        self.layers = nn.List([
            nn.SeqStack([ArtoriaDecoder(config, rngs=rngs) for _ in range(config.n_parallel)])
            for _ in range(config.n_depth)
        ])
        self.norm = nn.RMSNorm(config.d_model, config.epsilon, dtype='float32', axis_names=('embed',))
        self.eval_norm = nn.RMSNorm(config.d_model, config.epsilon, dtype='float32', axis_names=('embed',))
        self.head = nn.Linear(config.d_model, (70, config.vocab), bias=False, dtype=config.dtype, rngs=rngs, axis_names=('embed', 'slot', 'vocab'))
        self.eval_head = nn.Linear(config.d_model, 1, bias=True, dtype='float32', rngs=rngs, axis_names=('embed', 'eval'))
        self.policy_head = nn.Linear(config.d_model, config.n_moves, bias=False, dtype=config.dtype, rngs=rngs, axis_names=('embed', 'move'))
        self.rope = ArtoriaRoPE(config.head_dim)
        self.files = nn.Parameter( 0.02 * jax.random.normal(rngs(), (8, config.d_model)))
        self.ranks = nn.Parameter( 0.02 * jax.random.normal(rngs(), (8, config.d_model)))
        self.metadata_positions = nn.Parameter(0.02 * jax.random.normal(rngs(), (6, config.d_model)))
        self.k = config.n_parallel

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ArtoriaCache | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # ids: [B, T, 70] -> [B, T, 70, C]
        if ids.ndim != 3 or ids.shape[-1] != 70:
            raise ValueError('ids must have shape [batch, sequence, 70]')
        x = jax.checkpoint(self.wte)(ids)
        B, T, S, C = x.shape
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        layer_idx = jnp.asarray(0, dtype='uint32')

        coord = (self.ranks[:, None, :] + self.files[None, :, :]).reshape(64, -1)
        slot_positions = jnp.concatenate((coord, self.metadata_positions.value), axis=0)
        x = x + slot_positions.astype(x.dtype)[None, None]
        x = x.reshape(B * T, S, C)
        for spatial_layer in self.spatial_layers:
            x = jax.checkpoint(spatial_layer)(x)
        x = x.mean(axis=1).reshape(B, T, C)
        if mask is None and cache is None:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]
        elif mask is not None and mask.ndim == 3:
            mask = mask[:, None]
        def fwd_layer(layer, carry, z):
            x, layer_idx = carry
            x = jax.checkpoint(layer)(z, mask, position_embedding, cache, layer_idx) + x
            return (x, layer_idx + 1), None

        for layer in self.layers:
            z = x
            (x, layer_idx), _ = jax.checkpoint(layer, static_argnums=0)(fwd_layer, (x, layer_idx), z)
            x = x / self.k

        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(self.head)(x)

        ev = jax.checkpoint(self.eval_norm)(x)
        ev_logits = jax.checkpoint(self.eval_head)(ev)
        policy_logits = jax.checkpoint(self.policy_head)(x)

        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits, ev_logits, policy_logits


__all__ = ['Artoria']
