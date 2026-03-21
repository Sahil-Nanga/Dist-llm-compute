from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)



def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Root-Mean-Square Layer Normalisation (as used in LLaMA).
    x:      (..., hidden_dim)
    weight: (hidden_dim,)
    """
    ms  = np.mean(x * x, axis=-1, keepdims=True)
    x_n = x * np.reciprocal(np.sqrt(ms + eps))
    return x_n * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True) 
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)



def _build_rope_cache(seq_len: int, head_dim: int,
                      base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute (cos, sin) tables for RoPE up to seq_len positions.
    Returns cos, sin each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    theta = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    t     = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, theta)     
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply RoPE to query or key tensors.
    x:   (batch, seq_len, n_heads, head_dim)
    cos/sin: (seq_len, head_dim//2)
    """
    half  = x.shape[-1] // 2
    x1    = x[..., :half]
    x2    = x[..., half:]
    c = cos[np.newaxis, :, np.newaxis, :]
    s = sin[np.newaxis, :, np.newaxis, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)



class TransformerBlock:
    """
    One LLaMA-style transformer block (attention + SwiGLU FFN).

    Weights expected in the dict (prefix = "blk.{i}."):
      attn_norm.weight  (hidden,)
      attn_q.weight     (n_heads * head_dim, hidden)
      attn_k.weight     (n_kv_heads * head_dim, hidden)
      attn_v.weight     (n_kv_heads * head_dim, hidden)
      attn_output.weight (hidden, n_heads * head_dim)
      ffn_norm.weight   (hidden,)
      ffn_gate.weight   (intermediate, hidden)
      ffn_up.weight     (intermediate, hidden)
      ffn_down.weight   (hidden, intermediate)
    """

    def __init__(self, layer_idx: int, weights: Dict[str, np.ndarray],
                 n_heads: int, n_kv_heads: int, rope_base: float = 10000.0):
        self.idx        = layer_idx
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_base  = rope_base

        p = f"blk.{layer_idx}."
        self.W_attn_norm = weights[f"{p}attn_norm.weight"]
        self.W_q         = weights[f"{p}attn_q.weight"]
        self.W_k         = weights[f"{p}attn_k.weight"]
        self.W_v         = weights[f"{p}attn_v.weight"]
        self.W_o         = weights[f"{p}attn_output.weight"]
        self.W_ffn_norm  = weights[f"{p}ffn_norm.weight"]
        self.W_gate      = weights[f"{p}ffn_gate.weight"]
        self.W_up        = weights[f"{p}ffn_up.weight"]
        self.W_down      = weights[f"{p}ffn_down.weight"]

        self.b_q = weights.get(f"{p}attn_q.bias")
        self.b_k = weights.get(f"{p}attn_k.bias")
        self.b_v = weights.get(f"{p}attn_v.bias")
        self.b_o = weights.get(f"{p}attn_output.bias")

        self.hidden_dim  = self.W_attn_norm.shape[0]
        self.head_dim    = self.hidden_dim // n_heads
        self._rope_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._rope_len   = 0

    def _get_rope(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        if seq_len > self._rope_len:
            self._rope_cache = _build_rope_cache(seq_len, self.head_dim, self.rope_base)
            self._rope_len = seq_len
        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def _attention(self, x_norm: np.ndarray,
                   kv_cache: Optional[Tuple[np.ndarray, np.ndarray]],
                   positions: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        B, S, H = x_norm.shape
        hd = self.head_dim
        nq = self.n_heads
        nk = self.n_kv_heads

        Q = (x_norm @ self.W_q.T)
        if self.b_q is not None:
            Q = Q + self.b_q
        Q = Q.reshape(B, S, nq, hd)

        K = (x_norm @ self.W_k.T)
        if self.b_k is not None:
            K = K + self.b_k
        K = K.reshape(B, S, nk, hd)

        V = (x_norm @ self.W_v.T)
        if self.b_v is not None:
            V = V + self.b_v
        V = V.reshape(B, S, nk, hd)

        cos, sin = self._get_rope(S + (kv_cache[0].shape[1] if kv_cache else 0))
        pos_cos = cos[positions]  
        pos_sin = sin[positions]

        Q = apply_rope(Q, pos_cos, pos_sin)
        K = apply_rope(K, pos_cos, pos_sin)

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

        scale   = 1.0 / math.sqrt(hd)
        scores  = (Q @ K) * scale    

        if S > 1:
            mask = np.triu(np.full((S, T), -1e9, dtype=np.float32), k=T - S + 1)
            scores = scores + mask[np.newaxis, np.newaxis, :, :]

        attn = softmax(scores, axis=-1)         
        out  = (attn @ V).transpose(0, 2, 1, 3)  
        out  = out.reshape(B, S, nq * hd)         

        out = out @ self.W_o.T
        if self.b_o is not None:
            out = out + self.b_o
        return out, new_kv

    def forward(
        self,
        hidden: np.ndarray,
        positions: np.ndarray,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Args:
            hidden:    (batch, seq_len, hidden_dim)  float32
            positions: (seq_len,)                    int32
            kv_cache:  (K_past, V_past) or None
        Returns:
            (hidden_out, new_kv_cache)
        """
        x_norm = rms_norm(hidden, self.W_attn_norm)
        attn_out, new_kv = self._attention(x_norm, kv_cache, positions)
        hidden = hidden + attn_out  


        x_norm    = rms_norm(hidden, self.W_ffn_norm)
        gate_out  = silu(x_norm @ self.W_gate.T)  
        up_out    = x_norm @ self.W_up.T           
        ffn_out   = (gate_out * up_out) @ self.W_down.T 
        hidden    = hidden + ffn_out                

        return hidden, new_kv



class EmbeddingLayer:
    """Token embedding lookup. Lives on the master."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.embed = weights["token_embd.weight"]  
        log.info("EmbeddingLayer: vocab=%d, hidden=%d", *self.embed.shape)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (batch, seq_len) int64 → (batch, seq_len, hidden_dim)"""
        return self.embed[token_ids].astype(np.float32)


class OutputLayer:
    """Final RMSNorm + LM head. Lives on the master."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.norm   = weights["output_norm.weight"]
        self.lm_head = weights.get("output.weight")
        if self.lm_head is None:
            raise KeyError("output.weight not found; pass embedding matrix as output.weight")
        log.info("OutputLayer: vocab=%d", self.lm_head.shape[0])

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """hidden: (batch, seq_len, hidden_dim) → logits: (batch, seq_len, vocab_size)"""
        normed  = rms_norm(hidden, self.norm)
        logits  = normed @ self.lm_head.T   
        return logits



class LayerRangeEngine:
    """
    Executes forward pass for an assigned contiguous range of transformer blocks.

    This is the class instantiated on each worker after it receives its weights.
    It maintains a KV-cache dict keyed by request_id for concurrent requests
    (though inference is currently serialised, this is forward-compatible).
    """

    def __init__(
        self,
        layer_indices: List[int],
        weights_list: List[Dict[str, np.ndarray]],
        n_heads: int,
        n_kv_heads: int,
    ):
        assert len(layer_indices) == len(weights_list)
        self.layer_indices = layer_indices
        self.blocks = [
            TransformerBlock(li, w, n_heads, n_kv_heads)
            for li, w in zip(layer_indices, weights_list)
        ]
        self.kv_caches: Dict[int, List[Optional[Tuple[np.ndarray, np.ndarray]]]] = {}
        log.info(
            "LayerRangeEngine ready: layers %s..%s (%d blocks)",
            layer_indices[0], layer_indices[-1], len(self.blocks)
        )

    def forward(
        self,
        hidden: np.ndarray,
        positions: np.ndarray,
        request_id: int = 0,
        reset_cache: bool = False,
    ) -> np.ndarray:
        """
        Run all assigned blocks on the hidden state tensor.

        Args:
            hidden:     (batch, seq_len, hidden_dim)  float32
            positions:  (seq_len,)                    int32/int64
            request_id: used to separate KV caches for concurrent requests
            reset_cache: if True, clears the KV cache for this request_id
        Returns:
            updated hidden: (batch, seq_len, hidden_dim)
        """
        if reset_cache or request_id not in self.kv_caches:
            self.kv_caches[request_id] = [None] * len(self.blocks)

        caches = self.kv_caches[request_id]

        for i, block in enumerate(self.blocks):
            hidden, new_kv = block.forward(hidden, positions, kv_cache=caches[i])
            caches[i] = new_kv

        return hidden

    def clear_cache(self, request_id: int):
        self.kv_caches.pop(request_id, None)



def sample_token(
    logits: np.ndarray,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
) -> int:
    """
    Sample the next token from logits using temperature + top-k + top-p.

    logits: (vocab_size,)  — the final token position's logit vector.
    Returns a single integer token id.
    """
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
        sorted_idx  = np.argsort(-probs)
        cumsum      = np.cumsum(probs[sorted_idx])
        cutoff_mask = cumsum - probs[sorted_idx] > top_p
        probs[sorted_idx[cutoff_mask]] = 0.0
        probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))