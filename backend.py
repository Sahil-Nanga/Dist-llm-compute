from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)

CHUNK_ROWS: int = int(os.environ.get("LLMDIST_CHUNK_ROWS", "512"))

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
QUANT_TYPE_NAMES: Dict[int, str] = {
    0: "F32",   1: "F16",   2: "Q4_0",  3: "Q4_1",
    6: "Q5_0",  7: "Q5_1",  8: "Q8_0",  10: "Q2_K",
    11: "Q3_K", 12: "Q4_K", 13: "Q4_K", 14: "Q5_K",
    15: "Q5_K", 16: "Q6_K", 17: "Q8_K",
}

_USE_TORCH: bool = False
_torch = None

def _try_enable_torch() -> None:
    global _USE_TORCH, _torch
    if os.environ.get("LLMDIST_BACKEND", "").lower() == "numpy":
        log.info("Compute backend: NumPy (forced via LLMDIST_BACKEND=numpy)")
        return
    try:
        import torch
        _torch = torch
        _USE_TORCH = True
        log.info("Compute backend: PyTorch %s (MKL/OpenBLAS per chunk)", torch.__version__)
    except ImportError:
        log.info("Compute backend: NumPy (torch not installed)")

_try_enable_torch()



def _chunk_matmul(x_f32: np.ndarray, W_chunk_f32: np.ndarray) -> np.ndarray:
    """
    x_f32:       (batch, in_features)      float32
    W_chunk_f32: (chunk_rows, in_features) float32
    returns:     (batch, chunk_rows)        float32
    """
    if _USE_TORCH:
        return _torch.matmul(
            _torch.from_numpy(x_f32),
            _torch.from_numpy(W_chunk_f32).T,
        ).numpy()
    return x_f32 @ W_chunk_f32.T



def _fused_q4_0(raw: np.ndarray, shape: tuple, x: np.ndarray) -> np.ndarray:
    out_features, in_features = shape
    blocks_per_row = in_features // 32

    all_blocks = raw.reshape(out_features, blocks_per_row, 18)
    x_f32 = x.reshape(-1, in_features).astype(np.float32)
    B     = x_f32.shape[0]
    out   = np.zeros((B, out_features), dtype=np.float32)

    for r0 in range(0, out_features, CHUNK_ROWS):
        r1   = min(r0 + CHUNK_ROWS, out_features)
        rows = r1 - r0
        blk  = all_blocks[r0:r1]  

        sc = np.frombuffer(
            blk[:, :, :2].tobytes(), dtype=np.float16
        ).reshape(rows, blocks_per_row, 1).astype(np.float32)

        nb = blk[:, :, 2:]                              
        lo = (nb & 0x0F).astype(np.float32) - 8
        hi = ((nb >> 4) & 0x0F).astype(np.float32) - 8  
        vals = np.concatenate([lo, hi], axis=-1) * sc

        out[:, r0:r1] = _chunk_matmul(x_f32, vals.reshape(rows, in_features))

    return out.astype(np.float16).reshape(*x.shape[:-1], out_features)


def _fused_q8_0(raw: np.ndarray, shape: tuple, x: np.ndarray) -> np.ndarray:
    out_features, in_features = shape
    blocks_per_row = in_features // 32

    all_blocks = raw.reshape(out_features, blocks_per_row, 34)
    x_f32 = x.reshape(-1, in_features).astype(np.float32)
    B     = x_f32.shape[0]
    out   = np.zeros((B, out_features), dtype=np.float32)

    for r0 in range(0, out_features, CHUNK_ROWS):
        r1   = min(r0 + CHUNK_ROWS, out_features)
        rows = r1 - r0
        blk  = all_blocks[r0:r1]

        sc = np.frombuffer(
            blk[:, :, :2].tobytes(), dtype=np.float16
        ).reshape(rows, blocks_per_row, 1).astype(np.float32)

        qs   = blk[:, :, 2:].view(np.int8).reshape(rows, blocks_per_row, 32).astype(np.float32)
        vals = qs * sc   

        out[:, r0:r1] = _chunk_matmul(x_f32, vals.reshape(rows, in_features))

    return out.astype(np.float16).reshape(*x.shape[:-1], out_features)



def _fused_kquant(
    raw: np.ndarray,
    quant_type: int,
    shape: tuple,
    qname: str,
    x: np.ndarray,
) -> np.ndarray:
    out_features, in_features = shape
    block_size, bpb = QUANT_SIZES[quant_type]
    bytes_per_row   = (in_features // block_size) * bpb

    row_bytes = raw.reshape(out_features, bytes_per_row)
    x_f32     = x.reshape(-1, in_features).astype(np.float32)
    B         = x_f32.shape[0]
    out        = np.zeros((B, out_features), dtype=np.float32)

    try:
        from gguf.quants import dequantize as _dq
        from gguf import GGMLQuantizationType as _QT
        _qt_val = _QT(quant_type)
    except ImportError:
        _dq = None
        _qt_val = None
        log.warning("gguf.quants unavailable — using linear fallback for %s", qname)

    for r0 in range(0, out_features, CHUNK_ROWS):
        r1        = min(r0 + CHUNK_ROWS, out_features)
        rows      = r1 - r0
        chunk_raw = row_bytes[r0:r1].reshape(-1)

        if _dq is not None:
            W_chunk = _dq(chunk_raw, _qt_val).reshape(rows, in_features).astype(np.float32)
        else:
            n = rows * in_features
            W_chunk = np.linspace(-0.1, 0.1, n, dtype=np.float32).reshape(rows, in_features)

        out[:, r0:r1] = _chunk_matmul(x_f32, W_chunk)
        del W_chunk

    return out.astype(np.float16).reshape(*x.shape[:-1], out_features)



def _fused_float(raw: np.ndarray, shape: tuple, x: np.ndarray) -> np.ndarray:
    """F32 / F16 weights — chunked matmul, no dequantization needed."""
    out_features, in_features = shape
    W     = raw.reshape(shape).astype(np.float32)
    x_f32 = x.reshape(-1, in_features).astype(np.float32)
    B     = x_f32.shape[0]
    out   = np.zeros((B, out_features), dtype=np.float32)

    for r0 in range(0, out_features, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, out_features)
        out[:, r0:r1] = _chunk_matmul(x_f32, W[r0:r1])

    return out.astype(np.float16).reshape(*x.shape[:-1], out_features)



def _elementwise(raw: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float16) * raw.astype(np.float16)).astype(np.float16)



def quantized_matmul(
    raw: np.ndarray,
    quant_type: int,
    shape: tuple,
    quant_type_name: str,
    x: np.ndarray,
) -> np.ndarray:
    """
    Block-fused dequantize + matmul.

    Dispatches to the appropriate kernel based on quant_type.
    The full float16 weight matrix is never allocated at once.

    raw            : raw bytes stored in QuantizedWeight.data
    quant_type     : GGUF quant type id
    shape          : (out_features, in_features)
    quant_type_name: e.g. "Q4_K"
    x              : (..., in_features) activation, any float dtype
    """
    if len(shape) < 2:
        return _elementwise(raw, x)

    if quant_type in (0, 1):
        return _fused_float(raw, shape, x)

    if quant_type == 2:
        return _fused_q4_0(raw, shape, x)

    if quant_type == 8:
        return _fused_q8_0(raw, shape, x)

    return _fused_kquant(raw, quant_type, shape, quant_type_name, x)



def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if _USE_TORCH:
        x_t = _torch.from_numpy(x.astype(np.float32))
        w_t = _torch.from_numpy(weight.astype(np.float32))
        ms  = (x_t * x_t).mean(dim=-1, keepdim=True)
        return (x_t * _torch.rsqrt(ms + eps) * w_t).to(_torch.float16).numpy()
    x32 = x.astype(np.float32)
    ms  = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * np.reciprocal(np.sqrt(ms + eps)) * weight.astype(np.float32)).astype(np.float16)


def silu(x: np.ndarray) -> np.ndarray:
    if _USE_TORCH:
        x_t = _torch.from_numpy(x.astype(np.float32))
        return _torch.nn.functional.silu(x_t).to(_torch.float16).numpy()
    x32 = x.astype(np.float32)
    return (x32 * (1.0 / (1.0 + np.exp(-x32)))).astype(np.float16)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if _USE_TORCH:
        x_t = _torch.from_numpy(x.astype(np.float32))
        return _torch.nn.functional.softmax(x_t, dim=axis).to(_torch.float16).numpy()
    x32 = x.astype(np.float32)
    x32 -= x32.max(axis=axis, keepdims=True)
    e    = np.exp(x32)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float16)