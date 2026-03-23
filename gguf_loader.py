from __future__ import annotations

import gc
import struct
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

GGUF_MAGIC = 0x46554747
GGUF_ALIGN = 32

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
    11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K",
}
@dataclass
class TensorInfo:
    name:       str
    shape:      tuple
    quant_type: int
    offset:     int
    n_elements: int
    byte_size:  int


class GGUFIndex:
    def __init__(self, path: str):
        self.path = path
        self.tensors: Dict[str, TensorInfo] = {}
        self._parse()

    @staticmethod
    def _ru8(f):  return struct.unpack("<B", f.read(1))[0]
    @staticmethod
    def _ri8(f):  return struct.unpack("<b", f.read(1))[0]
    @staticmethod
    def _ru16(f): return struct.unpack("<H", f.read(2))[0]
    @staticmethod
    def _ri16(f): return struct.unpack("<h", f.read(2))[0]
    @staticmethod
    def _ru32(f): return struct.unpack("<I", f.read(4))[0]
    @staticmethod
    def _ri32(f): return struct.unpack("<i", f.read(4))[0]
    @staticmethod
    def _rf32(f): return struct.unpack("<f", f.read(4))[0]
    @staticmethod
    def _ru64(f): return struct.unpack("<Q", f.read(8))[0]
    @staticmethod
    def _ri64(f): return struct.unpack("<q", f.read(8))[0]
    @staticmethod
    def _rf64(f): return struct.unpack("<d", f.read(8))[0]
    @staticmethod
    def _rbool(f): return bool(struct.unpack("<B", f.read(1))[0])
    @staticmethod
    def _rstr(f) -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")

    def _rval(self, f, vtype: int):
        dispatch = {
            0: self._ru8,  1: self._ri8,  2: self._ru16, 3: self._ri16,
            4: self._ru32, 5: self._ri32, 6: self._rf32, 7: self._rbool,
            8: self._rstr, 10: self._ru64, 11: self._ri64, 12: self._rf64,
        }
        if vtype == 9:
            et = self._ru32(f)
            n = self._ru64(f)
            return [self._rval(f, et) for _ in range(n)]
        fn = dispatch.get(vtype)
        if fn is None:
            raise ValueError(f"Unknown GGUF value type: {vtype}")
        return fn(f)

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = self._ru32(f)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: {self.path}")
            _ver = self._ru32(f)
            n_tensors = self._ru64(f)
            n_kv = self._ru64(f)

            for _ in range(n_kv):
                self._rstr(f)
                vtype = self._ru32(f)
                self._rval(f, vtype)

            raw = []
            for _ in range(n_tensors):
                name = self._rstr(f)
                n_dims = self._ru32(f)
                dims = tuple(self._ru64(f) for _ in range(n_dims))
                qtype = self._ru32(f)
                offset = self._ru64(f)
                raw.append((name, dims, qtype, offset))

            pos = f.tell()
            data_start = (pos + GGUF_ALIGN - 1) & ~(GGUF_ALIGN - 1)

        for name, dims, qtype, rel_offset in raw:
            n_elems = 1
            for d in dims:
                n_elems *= d
            block_size, type_size = QUANT_SIZES.get(qtype, (1, 4))
            byte_size = int(n_elems * type_size / block_size)
            abs_offset = data_start + rel_offset
            self.tensors[name] = TensorInfo(
                name=name, shape=dims, quant_type=qtype,
                offset=abs_offset, n_elements=n_elems, byte_size=byte_size,
            )
        log.info("GGUFIndex: %d tensors indexed", len(self.tensors))

    def read_raw(self, name: str) -> Tuple[TensorInfo, np.ndarray]:
        info = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(info.offset)
            raw = f.read(info.byte_size)
        if info.quant_type == 0:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(info.shape).copy()
        elif info.quant_type == 1:
            arr = np.frombuffer(raw, dtype=np.float16).reshape(info.shape).copy()
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).copy()
        return info, arr


def _layer_tensor_names(layer_idx: int) -> List[str]:
    p = f"blk.{layer_idx}"
    return [
        f"{p}.attn_norm.weight", f"{p}.attn_q.weight",
        f"{p}.attn_k.weight",   f"{p}.attn_v.weight",
        f"{p}.attn_output.weight", f"{p}.ffn_norm.weight",
        f"{p}.ffn_gate.weight", f"{p}.ffn_up.weight",
        f"{p}.ffn_down.weight", f"{p}.attn_q.bias",
        f"{p}.attn_k.bias",     f"{p}.attn_v.bias",
        f"{p}.attn_output.bias",
    ]


EMBEDDING_NAMES = ["token_embd.weight"]
OUTPUT_NAMES = ["output_norm.weight", "output.weight"]


def build_index(model_path: str) -> GGUFIndex:
    return GGUFIndex(model_path)


def load_layer_streaming(index: GGUFIndex, layer_idx: int) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name in _layer_tensor_names(layer_idx):
        if name not in index.tensors:
            continue
        info, arr = index.read_raw(name)
        result[name] = arr
        result[f"__shape__{name}"] = list(info.shape)
        result[f"__qtype__{name}"] = info.quant_type
        log.debug("  <- %s  %.1f MiB", name, info.byte_size / (1024 ** 2))

    if not any(k for k in result if not k.startswith("__")):
        available = [k for k in index.tensors if k.startswith(f"blk.{layer_idx}.")]
        raise RuntimeError(f"No tensors found for layer {layer_idx}. Available: {available}")

    total_mib = sum(
        a.nbytes for k, a in result.items()
        if not k.startswith("__")
    ) / (1024 ** 2)
    log.info("Layer %d: %d tensors, %.1f MiB", layer_idx, len(result), total_mib)
    return result


def load_embedding_weights(model_path: str) -> Dict[str, np.ndarray]:
    idx = build_index(model_path)
    result = {}
    for name in EMBEDDING_NAMES:
        if name in idx.tensors:
            info, arr = idx.read_raw(name)
            result[name] = arr
            result[f"__shape__{name}"]  = list(info.shape)
            result[f"__qtype__{name}"]  = info.quant_type   
            log.info("Embedding %s shape=%s qtype=%d", name, info.shape, info.quant_type)
    if not result:
        raise RuntimeError(f"No embedding tensors in {model_path}")
    return result


def load_output_weights(model_path: str) -> Dict[str, np.ndarray]:
    idx = build_index(model_path)
    result = {}
    for name in OUTPUT_NAMES:
        if name in idx.tensors:
            info, arr = idx.read_raw(name)
            result[name] = arr
            result[f"__shape__{name}"] = list(info.shape)
            result[f"__qtype__{name}"] = info.quant_type
            log.info("Output %s shape=%s qtype=%d", name, info.shape, info.quant_type)
    return result


def load_layers_range(
    model_path: str,
    start_layer: int,
    end_layer: int,
) -> List[Dict[str, np.ndarray]]:
    idx = build_index(model_path)
    result = []
    for li in range(start_layer, end_layer):
        result.append(load_layer_streaming(idx, li))
        gc.collect()
    return result


def iter_layers(
    model_path: str,
    layer_indices: List[int],
) -> Generator[Tuple[int, Dict[str, np.ndarray]], None, None]:
    idx = build_index(model_path)
    for layer_idx in layer_indices:
        weights = load_layer_streaming(idx, layer_idx)
        yield layer_idx, weights
        del weights
        gc.collect()