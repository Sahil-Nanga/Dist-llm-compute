from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)



def _get_reader(path: str):
    """
    Return a gguf.GGUFReader for the given path.
    Lazy import so that the package is only required on nodes that actually
    load weights (i.e., all nodes — master and workers).
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        raise ImportError(
            "The 'gguf' package is required to load model weights.\n"
            "Install it with:  pip install gguf"
        )
    return GGUFReader(path, mode="r")


def _tensor_to_numpy(tensor) -> np.ndarray:
    """
    Convert a gguf.ReaderTensor to a float32 numpy array.
    GGUFReader already dequantises the data when you call .data.
    """
    data = tensor.data
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data



def _layer_tensor_names(layer_idx: int) -> List[str]:
    """
    Return the expected GGUF tensor names for transformer block `layer_idx`.
    Covers LLaMA-style architecture (blk.N.xxx).
    """
    prefix = f"blk.{layer_idx}"
    return [
        f"{prefix}.attn_norm.weight",
        f"{prefix}.attn_q.weight",
        f"{prefix}.attn_k.weight",
        f"{prefix}.attn_v.weight",
        f"{prefix}.attn_output.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.ffn_gate.weight",
        f"{prefix}.ffn_up.weight",
        f"{prefix}.ffn_down.weight",
        f"{prefix}.attn_q.bias",
        f"{prefix}.attn_k.bias",
        f"{prefix}.attn_v.bias",
        f"{prefix}.attn_output.bias",
    ]

EMBEDDING_TENSOR_NAMES = ["token_embd.weight"]
OUTPUT_TENSOR_NAMES    = ["output_norm.weight", "output.weight"]



def load_embedding_weights(model_path: str) -> Dict[str, np.ndarray]:
    """
    Load the token embedding matrix from the GGUF file.
    Returns {tensor_name: float32_array}.
    """
    reader = _get_reader(model_path)
    result: Dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.name in EMBEDDING_TENSOR_NAMES:
            result[tensor.name] = _tensor_to_numpy(tensor)
            log.debug("Loaded embedding tensor %s, shape=%s", tensor.name, result[tensor.name].shape)
    if not result:
        raise RuntimeError(f"No embedding tensors found in {model_path}")
    return result


def load_output_weights(model_path: str) -> Dict[str, np.ndarray]:
    """
    Load the final LayerNorm + LM head weights from the GGUF file.
    Returns {tensor_name: float32_array}.
    """
    reader = _get_reader(model_path)
    result: Dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.name in OUTPUT_TENSOR_NAMES:
            result[tensor.name] = _tensor_to_numpy(tensor)
            log.debug("Loaded output tensor %s, shape=%s", tensor.name, result[tensor.name].shape)
    return result


def load_layer_weights(model_path: str, layer_idx: int) -> Dict[str, np.ndarray]:
    """
    Load all weight tensors for transformer block `layer_idx`.
    Returns {tensor_name: float32_array}.

    Raises RuntimeError if no tensors are found for the requested layer.
    """
    expected = set(_layer_tensor_names(layer_idx))
    reader   = _get_reader(model_path)
    result: Dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.name in expected:
            result[tensor.name] = _tensor_to_numpy(tensor)
    if not result:
        raise RuntimeError(
            f"No tensors found for layer {layer_idx} in {model_path}. "
            f"Expected names like: {list(expected)[:3]} ..."
        )
    log.debug("Loaded %d tensor(s) for layer %d", len(result), layer_idx)
    return result


def load_layers_range(
    model_path: str,
    start_layer: int,
    end_layer: int, 
) -> List[Dict[str, np.ndarray]]:
    """
    Efficiently load a contiguous range of transformer layers in a single pass
    over the file (avoids re-opening for each layer).

    Returns a list of weight dicts: result[i] corresponds to layer (start_layer + i).
    """
    target_names: Dict[str, int] = {}  
    for li in range(start_layer, end_layer):
        for name in _layer_tensor_names(li):
            target_names[name] = li

    reader = _get_reader(model_path)
    per_layer: Dict[int, Dict[str, np.ndarray]] = {li: {} for li in range(start_layer, end_layer)}

    for tensor in reader.tensors:
        if tensor.name in target_names:
            li = target_names[tensor.name]
            per_layer[li][tensor.name] = _tensor_to_numpy(tensor)
            log.debug("  layer[%d] ← %s %s", li, tensor.name, per_layer[li][tensor.name].shape)

    result = [per_layer[li] for li in range(start_layer, end_layer)]
    loaded_count = sum(len(d) for d in result)
    log.info("Loaded %d weight tensors for layers %d–%d", loaded_count, start_layer, end_layer - 1)
    return result