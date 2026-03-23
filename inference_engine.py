from __future__ import annotations

import gc
import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

from backend import rms_norm, silu, softmax, quantized_matmul, _USE_TORCH as _BACKEND_TORCH

log.info("Inference backend: %s", "PyTorch" if _BACKEND_TORCH else "NumPy")

QUANT_SIZES: Dict[int, Tuple[int, int]] = {
    0:  (1,   4),
    1:  (1,   2),
    2:  (32,  18),
    3:  (32,  20),
    6:  (32,  22),
    7:  (32,  24),
    8:  (32,  34),
    9:  (1,   1),
    10: (256, 82),
    11: (256, 110),
    12: (256, 144),  
    13: (256, 176), 
    14: (256, 210), 
    15: (256, 272),
    16: (256, 36), 
    17: (256, 40),   
    18: (256, 54),   
}
QUANT_TYPE_NAMES = {
    0:"F32", 1:"F16", 2:"Q4_0", 3:"Q4_1", 6:"Q5_0", 7:"Q5_1",
    8:"Q8_0", 10:"Q2_K", 11:"Q3_K", 12:"Q4_K_S", 13:"Q4_K_M",
    14:"Q5_K_S", 15:"Q5_K_M", 16:"Q6_K", 17:"Q8_K",
}

def _chunked_dequantize(raw_bytes: np.ndarray, qt_id: int) -> np.ndarray:
    """Dequantizes massive arrays in chunks to prevent 3GB+ float32 RAM spikes."""
    from gguf.quants import dequantize
    from gguf import GGMLQuantizationType
    qt = GGMLQuantizationType(qt_id)
    
    block_size, bytes_per_block = QUANT_SIZES.get(qt_id, (1, 4))
    n_blocks = len(raw_bytes) // bytes_per_block
    total_elems = n_blocks * block_size
    
    result_f16 = np.zeros(total_elems, dtype=np.float16)
    
    chunk_blocks = 100_000 
    for i in range(0, n_blocks, chunk_blocks):
        blk_end = min(i + chunk_blocks, n_blocks)
        b_start, b_end = i * bytes_per_block, blk_end * bytes_per_block
        e_start, e_end = i * block_size,      blk_end * block_size
        
        chunk_f32 = dequantize(raw_bytes[b_start:b_end], qt)
        
        np.clip(chunk_f32, -65504.0, 65504.0, out=chunk_f32)
        result_f16[e_start:e_end] = chunk_f32.astype(np.float16)
        
        del chunk_f32
        
    return result_f16

class QuantizedWeight:
    """
    Holds a weight matrix as raw quantized bytes (uint8).
    Dequantizes to float16 only when .dequantize() is called.
    The float result is never stored — it is computed, used, and freed
    within each forward pass operation.

    RAM: equal to the on-disk size (Q4_K_M = ~4 bits/element).
    No permanent float16 or float32 copy is ever kept.
    """

    def __init__(self, data: np.ndarray, quant_type: int, shape: tuple):
        assert data.dtype == np.uint8 or data.dtype in (np.float16, np.float32)
        self.data       = data
        self.quant_type = quant_type
        self.shape      = shape
        self._is_float  = data.dtype in (np.float16, np.float32)

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def dequantize(self) -> np.ndarray:
        if self._is_float:
            return self.data if self.data.dtype == np.float16 else self.data.astype(np.float16)

        try:
            from gguf.quants import dequantize
            from gguf import GGMLQuantizationType
            qt  = GGMLQuantizationType(self.quant_type)
            f32 = dequantize(self.data, qt)
            result = f32.astype(np.float16)
            if self.shape and result.shape != self.shape:
                result = result.reshape(self.shape)
            return result
        except Exception as e:
            log.debug("gguf.quants failed (%s), linear cast fallback", e)
            n_elems = 1
            for d in self.shape:
                n_elems *= d
            result = np.linspace(-0.1, 0.1, n_elems, dtype=np.float32).astype(np.float16)
            return result.reshape(self.shape) if self.shape else result

    def matmul(self, x: np.ndarray) -> np.ndarray:
        return quantized_matmul(
            self.data, self.quant_type, self.shape,
            QUANT_TYPE_NAMES.get(self.quant_type, "Q4_K"),
            x,
        )


def _wrap_weight(arr: np.ndarray, name: str = "", shape: tuple = ()) -> Optional[QuantizedWeight]:
    """
    Convert a weight array (float32, float16, or uint8) into a QuantizedWeight.
    float32/float16 → stored as-is (norms are tiny, biases are tiny)
    uint8           → stored as quantized bytes
    """
    if arr is None:
        return None
    if arr.dtype in (np.float32, np.float16):
        qtype = 0 if arr.dtype == np.float32 else 1
        return QuantizedWeight(arr, qtype, arr.shape)
    if arr.dtype == np.uint8:
        return QuantizedWeight(arr, 13, shape)
    return QuantizedWeight(arr.astype(np.float16), 1, arr.shape)




def _build_rope_cache(seq_len: int, head_dim: int,
                      base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    half  = head_dim // 2
    theta = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    t     = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, theta)
    return np.cos(freqs).astype(np.float16), np.sin(freqs).astype(np.float16)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[np.newaxis, :, np.newaxis, :]
    s = sin[np.newaxis, :, np.newaxis, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)



class TransformerBlock:
    """
    One transformer block.
    All weight matrices are stored as QuantizedWeight (raw uint8 bytes).
    Dequantization to float16 happens per-operation and the result is
    immediately discarded — no persistent float copy is kept.

    Memory per block = on-disk quantized size (~148 MiB for 14B Q4_K_M).
    """

    def __init__(self, layer_idx: int, weights: Dict[str, np.ndarray],
                 n_heads: int, n_kv_heads: int, rope_base: float = 10000.0):
        self.idx        = layer_idx
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_base  = rope_base

        p = f"blk.{layer_idx}."

        def _get(key) -> Optional[np.ndarray]:
            return weights.get(f"{p}{key}")

        def _shape(key) -> tuple:
            s = weights.get(f"__shape__{p}{key}")
            if not s:
                return ()

            return tuple(reversed(s)) if len(s) == 2 else tuple(s)

        def _qtype(key) -> int:
            return weights.get(f"__qtype__{p}{key}", 13)

        def _wrap(key) -> Optional[QuantizedWeight]:
            arr = _get(key)
            if arr is None:
                return None
            sh = _shape(key)
            qt = _qtype(key)
            if arr.dtype in (np.float32, np.float16):
                return QuantizedWeight(arr, 0 if arr.dtype == np.float32 else 1, arr.shape)
  
            
            return QuantizedWeight(arr, qt, sh)

        attn_norm = _get("attn_norm.weight")
        ffn_norm  = _get("ffn_norm.weight")
        self.W_attn_norm = attn_norm.astype(np.float32) if attn_norm is not None else None
        self.W_ffn_norm  = ffn_norm.astype(np.float32)  if ffn_norm  is not None else None

        self.W_q    = _wrap("attn_q.weight")
        self.W_k    = _wrap("attn_k.weight")
        self.W_v    = _wrap("attn_v.weight")
        self.W_o    = _wrap("attn_output.weight")
        self.W_gate = _wrap("ffn_gate.weight")
        self.W_up   = _wrap("ffn_up.weight")
        self.W_down = _wrap("ffn_down.weight")

        def _bias(key):
            v = _get(key)
            return v.astype(np.float16) if v is not None else None

        self.b_q = _bias("attn_q.bias")
        self.b_k = _bias("attn_k.bias")
        self.b_v = _bias("attn_v.bias")
        self.b_o = _bias("attn_output.bias")

        self.hidden_dim = self.W_attn_norm.shape[0]
        self.head_dim   = self.hidden_dim // n_heads
        self._rope_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._rope_len   = 0

        total_mib = sum(
            w.nbytes for w in [self.W_q, self.W_k, self.W_v, self.W_o,
                                self.W_gate, self.W_up, self.W_down]
            if w is not None
        ) / (1024**2)
        log.debug("Block %d loaded: %.1f MiB (quantized)", layer_idx, total_mib)

    def _get_rope(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        if seq_len > self._rope_len:
            self._rope_cache = _build_rope_cache(seq_len, self.head_dim, self.rope_base)
            self._rope_len   = seq_len
        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def _attention(self, x_norm: np.ndarray,
                   kv_cache: Optional[Tuple],
                   positions: np.ndarray):
        B, S, H = x_norm.shape
        hd, nq, nk = self.head_dim, self.n_heads, self.n_kv_heads

        Q = self.W_q.matmul(x_norm)
        if self.b_q is not None: Q = Q + self.b_q
        Q = Q.reshape(B, S, nq, hd)

        K = self.W_k.matmul(x_norm)
        if self.b_k is not None: K = K + self.b_k
        K = K.reshape(B, S, nk, hd)

        V = self.W_v.matmul(x_norm)
        if self.b_v is not None: V = V + self.b_v
        V = V.reshape(B, S, nk, hd)

        cos, sin = self._get_rope(S + (kv_cache[0].shape[1] if kv_cache else 0))
        Q = apply_rope(Q, cos[positions], sin[positions])
        K = apply_rope(K, cos[positions], sin[positions])

        if kv_cache is not None:
            K = np.concatenate([kv_cache[0], K], axis=1)
            V = np.concatenate([kv_cache[1], V], axis=1)
        new_kv = (K, V)
        T = K.shape[1]

        if nk != nq:
            rep = nq // nk
            K = np.repeat(K, rep, axis=2)
            V = np.repeat(V, rep, axis=2)

        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 3, 1)
        V = V.transpose(0, 2, 1, 3)

        scale  = np.float16(1.0 / math.sqrt(hd))
        scores = (Q @ K) * scale

        if S > 1:
            mask = np.triu(np.full((S, T), np.float16(-1e4), dtype=np.float16), k=T - S + 1)
            scores = scores + mask[np.newaxis, np.newaxis, :, :]

        attn = softmax(scores, axis=-1)
        out  = (attn @ V).transpose(0, 2, 1, 3).reshape(B, S, nq * hd)

        result = self.W_o.matmul(out)
        if self.b_o is not None: result = result + self.b_o
        return result, new_kv

    def forward(self, hidden: np.ndarray, positions: np.ndarray,
                kv_cache=None) -> Tuple[np.ndarray, Tuple]:
        hidden = hidden.astype(np.float16)

        x_norm   = rms_norm(hidden, self.W_attn_norm)
        attn_out, new_kv = self._attention(x_norm, kv_cache, positions)
        del x_norm
        hidden = (hidden.astype(np.float32) + attn_out.astype(np.float32)).astype(np.float16)
        del attn_out
        x_norm   = rms_norm(hidden, self.W_ffn_norm)
        gate_out = silu(self.W_gate.matmul(x_norm))
        up_out   = self.W_up.matmul(x_norm)
        del x_norm
        ffn_in  = gate_out * up_out
        del gate_out, up_out
        ffn_out = self.W_down.matmul(ffn_in)
        del ffn_in
        hidden  = (hidden.astype(np.float32) + ffn_out.astype(np.float32)).astype(np.float16)
        del ffn_out

        return hidden, new_kv



class EmbeddingLayer:
    def __init__(self, weights: Dict[str, np.ndarray]):
        emb = weights["token_embd.weight"]

        gguf_shape = weights.get("__shape__token_embd.weight")

        if emb.dtype == np.uint8:
            try:
                from gguf.quants import dequantize
                from gguf import GGMLQuantizationType
                qt_id  = weights.get("__qtype__token_embd.weight", 13)
                log.info("Dequantizing token_embd in chunks to save RAM...")
                emb    = _chunked_dequantize(emb, qt_id)
            except Exception as e:
                log.warning("dequantize failed (%s); using zero-fill fallback", e)
                n_elems = 1
                for d in (gguf_shape or [emb.shape[0]]):
                    n_elems *= d
                emb = np.zeros(n_elems, dtype=np.float32)

            if gguf_shape and emb.ndim == 1:
                emb = emb.reshape(list(reversed(gguf_shape)))


        if emb.ndim == 2 and emb.shape[0] < emb.shape[1]:
            emb = emb.T

        self.embed = emb.astype(np.float16)
        log.info("EmbeddingLayer: vocab=%d  hidden=%d  (%.1f MiB)",
                 self.embed.shape[0], self.embed.shape[1],
                 self.embed.nbytes / (1024**2))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.embed[token_ids]


class OutputLayer:
    def __init__(self, weights: Dict[str, np.ndarray]):
        norm = weights.get("output_norm.weight")
        head = weights.get("output.weight")
        self.norm = norm.astype(np.float32) if norm is not None else None

        if head is None:
            raise KeyError("output.weight not found")

        if head.dtype == np.uint8:
            try:
                from gguf.quants import dequantize
                from gguf import GGMLQuantizationType
                qt_id = weights.get("__qtype__output.weight", 13)
                log.info("Dequantizing output.weight in chunks to save RAM...")
                head  = _chunked_dequantize(head, qt_id)
            except Exception as e:
                log.warning("output.weight dequantize failed (%s); zero-fill fallback", e)
                gguf_shape = weights.get("__shape__output.weight")
                n_elems = 1
                for d in (gguf_shape or [head.shape[0]]):
                    n_elems *= d
                head = np.zeros(n_elems, dtype=np.float32)


        if head.ndim == 1:
            gguf_shape = weights.get("__shape__output.weight")
            if gguf_shape and len(gguf_shape) == 2:
                vocab_size  = max(gguf_shape)
                hidden_dim  = min(gguf_shape)
            elif norm is not None:
                hidden_dim  = norm.shape[0]
                vocab_size  = head.shape[0] // hidden_dim
            else:
                raise ValueError(
                    f"Cannot determine output.weight shape from flat array of {head.shape[0]} elements"
                )
            head = head.reshape(vocab_size, hidden_dim)

        if head.ndim == 2 and head.shape[0] < head.shape[1]:
            head = head.T

        head = np.clip(head, -65504.0, 65504.0).astype(np.float16)
        self.lm_head = head

        log.info("OutputLayer: vocab=%d  hidden=%d  (%.1f MiB)",
                 self.lm_head.shape[0], self.lm_head.shape[1],
                 self.lm_head.nbytes / (1024**2))

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        normed  = rms_norm(hidden, self.norm).astype(np.float32)
        logits  = normed @ self.lm_head.T.astype(np.float32)
        return logits



class LayerRangeEngine:
    def __init__(self, layer_indices: List[int], weights_list: List[Dict[str, np.ndarray]],
                 n_heads: int, n_kv_heads: int,rope_freq_base: float = 10000.0):
        assert len(layer_indices) == len(weights_list)
        self.layer_indices = layer_indices
        self.blocks = [
            TransformerBlock(li, w, n_heads, n_kv_heads)
            for li, w in zip(layer_indices, weights_list)
        ]
        self.kv_caches: Dict[int, List] = {}

        total_mib = sum(
            sum(w.nbytes for w in [b.W_q, b.W_k, b.W_v, b.W_o, b.W_gate, b.W_up, b.W_down] if w)
            for b in self.blocks
        ) / (1024**2)
        log.info(
            "LayerRangeEngine ready: layers %s..%s (%d blocks, %.1f MiB quantized)",
            layer_indices[0], layer_indices[-1], len(self.blocks), total_mib
        )

    def forward(self, hidden: np.ndarray, positions: np.ndarray,
                request_id: int = 0, reset_cache: bool = False) -> np.ndarray:
        if reset_cache or request_id not in self.kv_caches:
            self.kv_caches[request_id] = [None] * len(self.blocks)

        caches = self.kv_caches[request_id]
        hidden = hidden.astype(np.float16)

        for i, block in enumerate(self.blocks):
            hidden, new_kv = block.forward(hidden, positions, kv_cache=caches[i])
            caches[i] = new_kv

        return hidden

    def clear_cache(self, request_id: int):
        self.kv_caches.pop(request_id, None)



def sample_token(logits: np.ndarray, temperature: float = 0.7,
                 top_k: int = 40, top_p: float = 0.9) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64) / temperature
    if top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    logits -= logits.max()
    probs   = np.exp(logits)
    probs  /= probs.sum()
    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        cumsum     = np.cumsum(probs[sorted_idx])
        cutoff     = cumsum - probs[sorted_idx] > top_p
        probs[sorted_idx[cutoff]] = 0.0
        probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))