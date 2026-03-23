"""
profiler.py — Dynamic Memory Profiler & Layer Assignment Engine.

Responsibilities:
  1. Parse GGUF binary headers to extract model architecture metadata
     without loading the full weight tensor data into memory.
  2. Query each device's available RAM via psutil.
  3. Calculate the exact memory footprint of each transformer layer.
  4. Run a greedy bin-packing algorithm to assign layers to devices,
     producing a deterministic {device_id: [layer_indices]} mapping.

This module contains ZERO hardcoded hardware assumptions. Every decision
is driven by real-time system data and model metadata.
"""

from __future__ import annotations

import struct
import logging
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from config import RAM_SAFETY_MARGIN, RAM_OVERHEAD_BYTES, MiB

log = logging.getLogger(__name__)


                                                                               
                                                                       

GGUF_MAGIC    = 0x46554747                            
GGUF_VERSION_1 = 1
GGUF_VERSION_2 = 2
GGUF_VERSION_3 = 3

                                        
class GGUFValueType:
    UINT8   = 0;  INT8   = 1
    UINT16  = 2;  INT16  = 3
    UINT32  = 4;  INT32  = 5
    FLOAT32 = 6;  BOOL   = 7
    STRING  = 8;  ARRAY  = 9
    UINT64  = 10; INT64  = 11
    FLOAT64 = 12

                                                    
                                            
GGUF_QUANT_SIZES: Dict[int, Tuple[int, int]] = {
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
                                                                                

@dataclasses.dataclass
class GGUFMetadata:
    """Parsed GGUF model metadata — no tensors, just the header info."""
    path: str
    architecture: str
    n_layers: int
    embedding_dim: int
    n_heads: int
    n_kv_heads: int
    intermediate_dim: int
    context_length: int
    vocab_size: int
    n_tensors: int
    dominant_quant_type: int                                                
    bytes_per_layer: int                                                       
    total_weight_bytes: int
    rope_freq_base: float                                                  

    def __repr__(self) -> str:
        return (
            f"GGUFMetadata(arch={self.architecture}, layers={self.n_layers}, "
            f"emb={self.embedding_dim}, heads={self.n_heads}, "
            f"bytes_per_layer={self.bytes_per_layer / MiB:.1f} MiB)"
        )


@dataclasses.dataclass
class DeviceProfile:
    """Snapshot of a device's available memory resources."""
    device_id: str
    total_ram_bytes: int
    available_ram_bytes: int
    usable_ram_bytes: int                                   

    @property
    def available_mib(self) -> float:
        return self.available_ram_bytes / MiB

    @property
    def usable_mib(self) -> float:
        return self.usable_ram_bytes / MiB


@dataclasses.dataclass
class LayerAssignment:
    """Result of the layer-assignment algorithm."""
    assignment: Dict[str, List[int]]                                         
    device_profiles: Dict[str, DeviceProfile]                             
    model_metadata: GGUFMetadata
    unassigned_layers: List[int]                                              
    pipeline_order: List[str]                                         

    @property
    def is_complete(self) -> bool:
        return len(self.unassigned_layers) == 0

    def summary(self) -> str:
        lines = ["Layer Assignment Summary:"]
        for dev in self.pipeline_order:
            layers = self.assignment.get(dev, [])
            mem = len(layers) * self.model_metadata.bytes_per_layer / MiB
            lines.append(
                f"  [{dev}] layers {layers[0] if layers else '-'}"
                f"–{layers[-1] if layers else '-'} "
                f"({len(layers)} blocks, ~{mem:.1f} MiB)"
            )
        if self.unassigned_layers:
            lines.append(f"  ⚠ UNASSIGNED: {self.unassigned_layers}")
        return "\n".join(lines)


                                                                                 

class GGUFParser:
    """
    Minimal streaming GGUF parser.

    Reads only the fixed-size header + KV metadata section.
    Does NOT load tensor data — this is intentional to keep memory use tiny
    even for 70B models.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = None
        self._version: int = 0
        self._kv: Dict[str, Any] = {}
        self._tensor_infos: List[Dict] = []

                                                                                

    def parse(self) -> GGUFMetadata:
        with open(self.path, "rb") as f:
            self._f = f
            self._read_header()
            self._read_kv_metadata()
            self._read_tensor_infos()

        return self._build_metadata()

                                                                                

    def _read_u8(self)  -> int: return struct.unpack("<B",  self._f.read(1))[0]
    def _read_i8(self)  -> int: return struct.unpack("<b",  self._f.read(1))[0]
    def _read_u16(self) -> int: return struct.unpack("<H",  self._f.read(2))[0]
    def _read_i16(self) -> int: return struct.unpack("<h",  self._f.read(2))[0]
    def _read_u32(self) -> int: return struct.unpack("<I",  self._f.read(4))[0]
    def _read_i32(self) -> int: return struct.unpack("<i",  self._f.read(4))[0]
    def _read_f32(self) -> float: return struct.unpack("<f", self._f.read(4))[0]
    def _read_u64(self) -> int: return struct.unpack("<Q",  self._f.read(8))[0]
    def _read_i64(self) -> int: return struct.unpack("<q",  self._f.read(8))[0]
    def _read_f64(self) -> float: return struct.unpack("<d", self._f.read(8))[0]
    def _read_bool(self) -> bool: return bool(self._read_u8())

    def _read_string(self) -> str:
        length = self._read_u64()
        return self._f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, vtype: int) -> Any:
        dispatch = {
            GGUFValueType.UINT8:   self._read_u8,
            GGUFValueType.INT8:    self._read_i8,
            GGUFValueType.UINT16:  self._read_u16,
            GGUFValueType.INT16:   self._read_i16,
            GGUFValueType.UINT32:  self._read_u32,
            GGUFValueType.INT32:   self._read_i32,
            GGUFValueType.FLOAT32: self._read_f32,
            GGUFValueType.BOOL:    self._read_bool,
            GGUFValueType.STRING:  self._read_string,
            GGUFValueType.UINT64:  self._read_u64,
            GGUFValueType.INT64:   self._read_i64,
            GGUFValueType.FLOAT64: self._read_f64,
        }
        if vtype == GGUFValueType.ARRAY:
            elem_type = self._read_u32()
            n_elems   = self._read_u64()
            return [self._read_value(elem_type) for _ in range(n_elems)]
        fn = dispatch.get(vtype)
        if fn is None:
            raise ValueError(f"Unknown GGUF value type: {vtype}")
        return fn()

                                                                                

    def _read_header(self):
        magic   = self._read_u32()
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a valid GGUF file (magic=0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
            )
        self._version   = self._read_u32()
        self._n_tensors = self._read_u64()
        self._n_kv      = self._read_u64()
        log.debug("GGUF v%d — %d tensors, %d KV pairs", self._version, self._n_tensors, self._n_kv)

    def _read_kv_metadata(self):
        for _ in range(self._n_kv):
            key   = self._read_string()
            vtype = self._read_u32()
            value = self._read_value(vtype)
            self._kv[key] = value
            log.debug("  KV: %s = %r", key, value if not isinstance(value, list) else f"[{len(value)} items]")

    def _read_tensor_infos(self):
        for _ in range(self._n_tensors):
            name   = self._read_string()
            n_dims = self._read_u32()
            dims   = [self._read_u64() for _ in range(n_dims)]
            ttype  = self._read_u32()                      
            offset = self._read_u64()                             
            self._tensor_infos.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

                                                                                

    def _get_kv(self, *keys, default=None) -> Any:
        """Try multiple KV keys (different model families use different prefixes)."""
        for k in keys:
            if k in self._kv:
                return self._kv[k]
        return default

    def _detect_architecture(self) -> str:
        arch = self._get_kv("general.architecture", default="unknown")
        return str(arch)

    def _get_n_layers(self, arch: str) -> int:
        return int(self._get_kv(
            f"{arch}.block_count",
            "llama.block_count", "falcon.block_count",
            "gpt2.block_count", "bloom.block_count",
            default=32
        ))

    def _get_embedding_dim(self, arch: str) -> int:
        return int(self._get_kv(
            f"{arch}.embedding_length",
            "llama.embedding_length",
            default=4096
        ))

    def _get_intermediate_dim(self, arch: str, emb_dim: int) -> int:
        val = self._get_kv(
            f"{arch}.feed_forward_length",
            "llama.feed_forward_length",
            default=None
        )
        return int(val) if val is not None else int(emb_dim * 2.6875)                       

    def _get_n_heads(self, arch: str) -> int:
        return int(self._get_kv(
            f"{arch}.attention.head_count",
            "llama.attention.head_count",
            default=32
        ))

    def _get_n_kv_heads(self, arch: str, n_heads: int) -> int:
        return int(self._get_kv(
            f"{arch}.attention.head_count_kv",
            "llama.attention.head_count_kv",
            default=n_heads
        ))

    def _get_context_length(self, arch: str) -> int:
        return int(self._get_kv(
            f"{arch}.context_length",
            "llama.context_length",
            default=4096
        ))

    def _get_vocab_size(self) -> int:
        val = self._get_kv("tokenizer.ggml.tokens", default=None)
        if isinstance(val, list):
            return len(val)
        return int(self._get_kv("llama.vocab_size", default=32000))

    def _dominant_quant_type(self) -> int:
        """Return the most common quantisation type used in transformer blocks."""
        from collections import Counter
                                                                                  
        block_tensors = [t for t in self._tensor_infos if t["name"].startswith("blk.")]
        if not block_tensors:
                                      
            block_tensors = self._tensor_infos
        if not block_tensors:
            return 1               
        counter = Counter(t["type"] for t in block_tensors)
        return counter.most_common(1)[0][0]

    def _calculate_bytes_per_layer(self, arch: str, emb_dim: int, intermediate_dim: int,
                                    n_heads: int, n_kv_heads: int, quant_type: int) -> int:
        """
        Estimate the memory cost of a single transformer block.

        LLaMA-style block contains:
          - attn_norm:   (emb_dim,)
          - attn_q:      (n_heads * head_dim, emb_dim)
          - attn_k:      (n_kv_heads * head_dim, emb_dim)
          - attn_v:      (n_kv_heads * head_dim, emb_dim)
          - attn_output: (emb_dim, n_heads * head_dim)
          - ffn_norm:    (emb_dim,)
          - ffn_gate:    (intermediate_dim, emb_dim)
          - ffn_up:      (intermediate_dim, emb_dim)
          - ffn_down:    (emb_dim, intermediate_dim)
        """
        head_dim = emb_dim // n_heads

                                       
        n_q = n_heads * head_dim * emb_dim
        n_k = n_kv_heads * head_dim * emb_dim
        n_v = n_kv_heads * head_dim * emb_dim
        n_o = emb_dim * (n_heads * head_dim)
        n_gate = intermediate_dim * emb_dim
        n_up   = intermediate_dim * emb_dim
        n_down = emb_dim * intermediate_dim
        n_norm = emb_dim * 2                                          

        weight_elements = n_q + n_k + n_v + n_o + n_gate + n_up + n_down
        norm_bytes = n_norm * 4                              

                                                       
        block_size, type_size = GGUF_QUANT_SIZES.get(quant_type, (1, 2))
        bpe = type_size / block_size                      

        weight_bytes = int(weight_elements * bpe)
        on_disk_bytes = weight_bytes + norm_bytes
        return on_disk_bytes

    def _total_weight_bytes(self) -> int:
        total = 0
        for ti in self._tensor_infos:
            n_elems = 1
            for d in ti["dims"]:
                n_elems *= d
            block_size, type_size = GGUF_QUANT_SIZES.get(ti["type"], (1, 4))
            total += int(n_elems * type_size / block_size)
        return total

    def _build_metadata(self) -> GGUFMetadata:
        arch         = self._detect_architecture()
        n_layers     = self._get_n_layers(arch)
        emb_dim      = self._get_embedding_dim(arch)
        n_heads      = self._get_n_heads(arch)
        n_kv_heads   = self._get_n_kv_heads(arch, n_heads)
        inter_dim    = self._get_intermediate_dim(arch, emb_dim)
        ctx_len      = self._get_context_length(arch)
        vocab_size   = self._get_vocab_size()
        quant_type   = self._dominant_quant_type()
        bpl          = self._calculate_bytes_per_layer(
                           arch, emb_dim, inter_dim, n_heads, n_kv_heads, quant_type)
        total_bytes  = self._total_weight_bytes()
        rope_base    = float(self._get_kv(f"{arch}.rope.freq_base", default=10000.0))

        return GGUFMetadata(
            path=str(self.path),
            architecture=arch,
            n_layers=n_layers,
            embedding_dim=emb_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            intermediate_dim=inter_dim,
            context_length=ctx_len,
            vocab_size=vocab_size,
            n_tensors=len(self._tensor_infos),
            dominant_quant_type=quant_type,
            bytes_per_layer=bpl,
            total_weight_bytes=total_bytes,
            rope_freq_base=rope_base,
        )


                                                                                

def get_local_device_profile(device_id: str = "master") -> DeviceProfile:
    """
    Snapshot the current system's memory using psutil.
    This is called on BOTH master and workers to get live RAM figures.
    """
    mem = psutil.virtual_memory()
    after_overhead = max(0, mem.available - RAM_OVERHEAD_BYTES)
    usable = int(after_overhead * RAM_SAFETY_MARGIN)
    return DeviceProfile(
        device_id=device_id,
        total_ram_bytes=mem.total,
        available_ram_bytes=mem.available,
        usable_ram_bytes=usable,
    )


def ram_bytes_from_report(report: dict) -> DeviceProfile:
    """Re-hydrate a DeviceProfile from a dict received over the network."""
    return DeviceProfile(
        device_id=report["device_id"],
        total_ram_bytes=report["total_ram_bytes"],
        available_ram_bytes=report["available_ram_bytes"],
        usable_ram_bytes=int(
            max(0, report["available_ram_bytes"] - RAM_OVERHEAD_BYTES) * RAM_SAFETY_MARGIN
        ),
    )


                                                                                

class LayerAssigner:
    """
    Greedy bin-packing layer assigner.

    Algorithm:
      1. Collect (device_id, usable_ram_bytes) for all devices, sorted
         so devices with MORE RAM appear first.
      2. Iterate through layers 0..N-1. For each layer, place it on the
         current device if it fits. Otherwise advance to the next device.
      3. Master always gets the first and/or last layers so it can handle
         embedding + LM-head without extra round-trips.
      4. Return the complete assignment plus a pipeline ordering.

    This runs entirely from real-time psutil + GGUF data — no hardcoding.
    """

    def __init__(self, metadata: GGUFMetadata, devices: Dict[str, DeviceProfile]):
        self.meta    = metadata
        self.devices = devices

    def assign(self) -> LayerAssignment:
        bpl         = self.meta.bytes_per_layer
        n_layers    = self.meta.n_layers
        assignment: Dict[str, List[int]] = {d: [] for d in self.devices}

        def sort_key(dev_id: str) -> Tuple[int, int]:
            priority = 0 if dev_id == "master" else 1
            return (priority, -self.devices[dev_id].usable_ram_bytes)

        ordered_devices = sorted(self.devices.keys(), key=sort_key)

        remaining: Dict[str, int] = {
            d: self.devices[d].usable_ram_bytes for d in ordered_devices
        }

        if "master" in remaining:
            vocab_elems = self.meta.vocab_size * self.meta.embedding_dim
            non_layer_bytes = vocab_elems * 4
            
            remaining["master"] -= non_layer_bytes
            
            log.info(
                "Reserved %.1f MiB on master for embedding/output matrices", 
                non_layer_bytes / (1024**2)
            )

        unassigned: List[int] = []
        dev_iter = iter(ordered_devices)
        current_dev: Optional[str] = next(dev_iter, None)

        for layer_idx in range(n_layers):
            placed = False
            while current_dev is not None:
                if remaining[current_dev] >= bpl:
                    assignment[current_dev].append(layer_idx)
                    remaining[current_dev] -= bpl
                    placed = True
                    break
                else:
                    current_dev = next(dev_iter, None)

            if not placed:
                unassigned.append(layer_idx)
                log.warning("Layer %d could not be assigned — insufficient RAM on all devices", layer_idx)

        pipeline_order = [d for d in ordered_devices if assignment[d]]

        return LayerAssignment(
            assignment=assignment,
            device_profiles=self.devices,
            model_metadata=self.meta,
            unassigned_layers=unassigned,
            pipeline_order=pipeline_order,
        )                                                                        

def parse_model(model_path: str) -> GGUFMetadata:
    """Parse GGUF metadata. Raises ValueError for non-GGUF files."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if path.suffix.lower() != ".gguf":
        raise ValueError(f"Expected a .gguf file, got: {path.suffix}")
    parser = GGUFParser(path)
    meta = parser.parse()
    log.info("Parsed model: %r", meta)
    return meta


def compute_layer_assignment(
    metadata: GGUFMetadata,
    device_profiles: Dict[str, DeviceProfile],
) -> LayerAssignment:
    """
    Given model metadata and a dict of live device profiles,
    return the optimal layer assignment.

    Args:
        metadata:        Output from parse_model().
        device_profiles: {device_id: DeviceProfile} — master + all connected workers.
    Returns:
        LayerAssignment with .is_complete flag and .summary() method.
    """
    assigner = LayerAssigner(metadata, device_profiles)
    result   = assigner.assign()
    log.info("\n%s", result.summary())
    return result