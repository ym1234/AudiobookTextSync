# Copyright © 2023 Apple Inc.

import base64
import gzip
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


_CACHE_CHUNK = 32 # positions to round the cache up to, see KVCache


class LayerCache:
    """one decoder layer's slice of the KVCache, see there for the layout"""

    def __init__(self, owner, shape, dtype):
        self._owner = owner
        self.k = mx.zeros(shape, dtype)
        self.v = mx.zeros(shape, dtype)
        self.cross = None

    def update(self, k, v):
        """write this step's keys/values at the write head and hand back the history including them"""
        start, end = self._owner.offset, self._owner.offset + k.shape[2]
        if end > self.k.shape[2]:
            self.k, self.v = self._owner.grow(self.k, end), self._owner.grow(self.v, end)
        self.k[:, :, start:end], self.v[:, :, start:end] = k, v
        return self.k[:, :, :end], self.v[:, :, :end]


class KVCache:
    """the decoder's cache, grown a chunk at a time rather than a step at a time

    Self attention keys/values are kept in the (batch, head, position, dim) layout the attention
    kernel wants, so a step writes a single slot rather than reallocating the whole history and
    transposing it again. Cross attention keys/values are a function of the audio alone, so they
    are computed when the cache is made and then never touched.

    The buffer is rounded up to a whole number of _CACHE_CHUNK positions instead of straight to the
    longest reachable sequence, so a short window doesn't hold memory for a long one.
    """

    def __init__(self, n_layers, batch, max_len, n_head, head_dim, dtype, initial):
        self.max_len = max_len
        self.layers = [LayerCache(self, (batch, n_head, self._room(initial), head_dim), dtype)
                       for _ in range(n_layers)]
        self.offset = 0

    def _room(self, end):
        """the buffer size that holds `end` positions: a whole number of chunks, capped at max_len"""
        assert end <= self.max_len, f"decoded past the {self.max_len} positions the cache can reach"
        return min(self.max_len, -(-end // _CACHE_CHUNK) * _CACHE_CHUNK)

    def grow(self, a, end):
        b, h, have, d = a.shape
        return mx.concatenate([a, mx.zeros((b, h, self._room(end) - have, d), a.dtype)], axis=2)

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def advance(self, n):
        self.offset += n

    @property
    def state(self):
        """every array in the cache, for mx.eval/mx.async_eval to bound the graph on"""
        return [a for l in self.layers for a in (l.k, l.v, *(l.cross or ()))]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.scale = (n_state // n_head) ** -0.5
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def _heads(self, x):
        """(batch, length, n_state) -> (batch, head, length, head_dim)"""
        return x.reshape(*x.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

    def kv(self, x):
        return mx.contiguous(self._heads(self.key(x))), mx.contiguous(self._heads(self.value(x)))

    def self_attention(self, x, mask=None, cache=None):
        k, v = self._heads(self.key(x)), self._heads(self.value(x))
        if cache is not None:
            k, v = cache.update(k, v)
        return self._attend(x, k, v, mask)[0]

    def cross_attention(self, x, kv, return_qk=False):
        return self._attend(x, *kv, return_qk=return_qk)

    def _attend(self, x, k, v, mask=None, return_qk=False):
        q = self._heads(self.query(x))
        if return_qk:
            qk = (q * self.scale) @ k.transpose(0, 1, 3, 2)
            if mask is not None:
                qk = qk + mask
            out = mx.softmax(qk, axis=-1, precise=True).astype(v.dtype) @ v
        else:
            qk, out = None, mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3)
        return self.out(out.reshape(*out.shape[:2], -1)), qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, cross_kv=None, mask=None, cache=None, return_qk=False):
        x = x + self.attn.self_attention(self.attn_ln(x), mask=mask, cache=cache)
        cross_qk = None
        if self.cross_attn:
            y, cross_qk = self.cross_attn.cross_attention(self.cross_attn_ln(x), cross_kv, return_qk=return_qk)
            x = x + y
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        return x, cross_qk


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = nn.LayerNorm(n_state)

    def __call__(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        assert x.shape[1:] == self._positional_embedding.shape, "incorrect audio shape"
        x = x + self._positional_embedding

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = nn.LayerNorm(n_state)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )
        self._cache_shape = (n_ctx, n_head, n_state // n_head)
        self._dtype = dtype

    def new_cache(self, batch, prompt_length):
        """size the cache to what a search can actually reach: the prompt plus the decode limit

        The searches in ats.model stop after n_ctx // 2 steps, so that plus the prompt is as far as
        the cache can ever be written; it starts out only big enough for the prompt and grows.
        """
        n_ctx, n_head, head_dim = self._cache_shape
        max_len = min(n_ctx, prompt_length + n_ctx // 2)
        return KVCache(len(self.blocks), batch, max_len, n_head, head_dim, self._dtype, prompt_length)

    def __call__(self, x, xa, kv_cache=None, return_qk=False):
        """
        x : mx.array, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : mx.array, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        return_qk : bool
            hand back the cross attention weights for alignment, at the cost of the slower unfused path
        """
        length = x.shape[-1]
        if kv_cache is None: # a new decode: the audio's keys/values are made here once, every step after reads them
            kv_cache = self.new_cache(x.shape[0], length)
            for block, layer in zip(self.blocks, kv_cache.layers):
                layer.cross = block.cross_attn.kv(xa)
        offset = kv_cache.offset
        # a causal mask is only meaningful for a multi token prompt, and one always starts from an
        # empty cache here; a single token step attends to the whole history and needs no mask
        assert offset == 0 or length == 1, "a multi token step into a warm cache would need its own mask"

        x = self.token_embedding(x) + self.positional_embedding[offset : offset + length]

        mask = self._mask[:length, :length] if length > 1 else None
        cross_qk = [None] * len(self.blocks)
        for e, block in enumerate(self.blocks):
            x, cross_qk[e] = block(x, kv_cache[e].cross, mask=mask, cache=kv_cache[e], return_qk=return_qk)
        kv_cache.advance(length)

        x = self.ln(x)
        return self.token_embedding.as_linear(x), kv_cache, cross_qk


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    def set_alignment_heads(self, dump: Union[bytes, np.ndarray]):
        if isinstance(dump, np.ndarray):
            self.alignment_heads = mx.array(dump)
        elif isinstance(dump, bytes):
            array = np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
            mask = array.reshape(self.dims.n_text_layer, self.dims.n_text_head)
            self.alignment_heads = mx.array(np.asarray(mask.nonzero()).T)
        else:
            raise ValueError(
                f"Invalid type for `dump`: {type(dump)}. Expected a np.ndarray or base85-encoded bytes containing"
                " alignment_head information"
            )

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk(self, mel, tokens):
        logits, _, cross_qk = self.decoder(tokens, self.encoder(mel), return_qk=True)
        return logits, cross_qk

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))[0]

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)


def load_model(path, dtype: mx.Dtype = mx.float16) -> Whisper:
    """Load an mlx-community style whisper checkpoint (config.json + weights.{npz,safetensors})."""
    path = Path(path)
    with open(path / "config.json", "r") as f:
        config = json.load(f)

    quantization = config.pop("quantization", None)
    known = {f.name for f in fields(ModelDimensions)}
    dims = ModelDimensions(**{k: v for k, v in config.items() if k in known})

    weights_path = path / "weights.safetensors"
    if not weights_path.exists():
        weights_path = path / "weights.npz"
    weights = mx.load(str(weights_path))

    model = Whisper(dims, dtype)
    if quantization is not None:
        # scales/biases live next to the packed weights, only quantize what was quantized
        nn.quantize(model, **quantization,
                    class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights)
    else:
        weights = {k: v.astype(dtype) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
                   for k, v in weights.items()}

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
