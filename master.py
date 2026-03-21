from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    MASTER_API_PORT, MASTER_API_HOST,
    ZMQ_CONTROL_PORT, ZMQ_WEIGHT_PORT, ZMQ_PIPELINE_PORT, ZMQ_COLLECT_PORT,
    CMD_HELLO, CMD_RAM_REPORT, CMD_ASSIGN_LAYERS, CMD_SET_NEXT_HOP,
    CMD_START_INFERENCE, CMD_RESET, CMD_ACK,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
    RAM_SAFETY_MARGIN, MiB,
)
from networking import (
    ZMQContext, ControlChannel, WeightChannel, PipelineChannel,
    DeviceDiscovery, WorkerServiceInfo, get_local_ip, TensorTransport,
)
from profiler import (
    parse_model, compute_layer_assignment,
    get_local_device_profile, ram_bytes_from_report,
    GGUFMetadata, LayerAssignment, DeviceProfile,
)
from gguf_loader import (
    load_embedding_weights, load_output_weights, load_layers_range,
)
from inference_engine import (
    EmbeddingLayer, OutputLayer, LayerRangeEngine, sample_token,
)
from tokenizer import GGUFTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("master")


class MasterState:
    def __init__(self):
        self.workers: Dict[str, WorkerServiceInfo] = {}
        self.worker_profiles: Dict[str, DeviceProfile] = {}
        self.worker_ctrl: Dict[str, ControlChannel] = {}
        self.worker_weight: Dict[str, WeightChannel] = {}
        self.assignment: Optional[LayerAssignment] = None
        self.model_metadata: Optional[GGUFMetadata] = None
        self.model_path: Optional[str] = None
        self.pipeline_order: List[str] = []
        self.embed_layer: Optional[EmbeddingLayer] = None
        self.output_layer: Optional[OutputLayer] = None
        self.master_engine: Optional[LayerRangeEngine] = None
        self.tokenizer: Optional[GGUFTokenizer] = None
        self.zmq: Optional[ZMQContext] = None
        self.pipe_out: Optional[PipelineChannel] = None
        self.pipe_collect: Optional[PipelineChannel] = None
        self.status: str = "idle"
        self.error: Optional[str] = None
        self._lock = asyncio.Lock()

    def to_status_dict(self) -> dict:
        return {
            "status":        self.status,
            "error":         self.error,
            "model_loaded":  self.model_path is not None,
            "model_path":    self.model_path,
            "n_workers":     len(self.workers),
            "pipeline_ready": len(self.pipeline_order) > 0,
            "master_ip":     get_local_ip(),
        }


STATE = MasterState()


app = FastAPI(title="DistributedLLM Master", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index = _frontend_dir / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Dashboard not found — place frontend/ next to master.py</h1>")


class LoadModelRequest(BaseModel):
    model_path: str

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int   = DEFAULT_MAX_NEW_TOKENS
    temperature:   float  = DEFAULT_TEMPERATURE
    top_p:         float  = DEFAULT_TOP_P
    top_k:         int    = DEFAULT_TOP_K

class DiscoverRequest(BaseModel):
    timeout: float = 15.0


@app.get("/api/status")
async def api_status():
    return STATE.to_status_dict()


@app.get("/api/workers")
async def api_workers():
    workers = []
    for dev_id, info in STATE.workers.items():
        profile = STATE.worker_profiles.get(dev_id)
        assignment = []
        if STATE.assignment:
            assignment = STATE.assignment.assignment.get(dev_id, [])
        workers.append({
            "device_id":          dev_id,
            "host":               info.host,
            "port":               info.port,
            "total_ram_bytes":    profile.total_ram_bytes  if profile else None,
            "available_ram_bytes":profile.available_ram_bytes if profile else None,
            "usable_ram_bytes":   profile.usable_ram_bytes if profile else None,
            "assigned_layers":    assignment,
            "n_layers":           len(assignment),
        })

    master_profile = get_local_device_profile("master")
    master_assignment = []
    if STATE.assignment:
        master_assignment = STATE.assignment.assignment.get("master", [])
    workers.insert(0, {
        "device_id":           "master",
        "host":                get_local_ip(),
        "port":                MASTER_API_PORT,
        "total_ram_bytes":     master_profile.total_ram_bytes,
        "available_ram_bytes": master_profile.available_ram_bytes,
        "usable_ram_bytes":    master_profile.usable_ram_bytes,
        "assigned_layers":     master_assignment,
        "n_layers":            len(master_assignment),
        "is_master":           True,
    })
    return {"workers": workers}


@app.post("/api/discover")
async def api_discover(req: DiscoverRequest, background: BackgroundTasks):
    STATE.status = "discovering"
    background.add_task(_discovery_task, req.timeout)
    return {"message": f"Discovery started (timeout={req.timeout}s)"}


@app.get("/api/models")
async def api_models(models_dir: str = "./models"):
    p = Path(models_dir)
    if not p.exists():
        return {"models": [], "models_dir": str(p.resolve())}
    gguf_files = sorted(p.glob("**/*.gguf"))
    return {
        "models":     [str(f) for f in gguf_files],
        "models_dir": str(p.resolve()),
    }


@app.post("/api/load-model")
async def api_load_model(req: LoadModelRequest):
    STATE.status = "loading_model"
    STATE.error  = None
    try:
        meta = parse_model(req.model_path)
        STATE.model_metadata = meta
        STATE.model_path     = req.model_path

        profiles: Dict[str, DeviceProfile] = {
            "master": get_local_device_profile("master")
        }
        profiles.update(STATE.worker_profiles)

        if not profiles:
            raise HTTPException(400, "No devices profiled yet — run /api/discover first")

        assignment = compute_layer_assignment(meta, profiles)
        STATE.assignment     = assignment
        STATE.pipeline_order = assignment.pipeline_order

        try:
            STATE.tokenizer = GGUFTokenizer.from_gguf(req.model_path)
            tok_info = {
                "tokenizer_type":    STATE.tokenizer.model_type,
                "vocab_size":        STATE.tokenizer.vocab_size,
                "bos_id":            STATE.tokenizer.bos_id,
                "eos_id":            STATE.tokenizer.eos_id,
                "has_chat_template": STATE.tokenizer.has_chat_template(),
            }
            log.info("Tokenizer loaded: %r", STATE.tokenizer)
        except Exception as tok_err:
            log.warning("Tokenizer loading failed (will use byte fallback): %s", tok_err)
            tok_info = {"tokenizer_type": "byte_fallback", "error": str(tok_err)}

        STATE.status = "model_ready"
        return {
            "architecture":     meta.architecture,
            "n_layers":         meta.n_layers,
            "embedding_dim":    meta.embedding_dim,
            "n_heads":          meta.n_heads,
            "n_kv_heads":       meta.n_kv_heads,
            "intermediate_dim": meta.intermediate_dim,
            "vocab_size":       meta.vocab_size,
            "bytes_per_layer":  meta.bytes_per_layer,
            "total_weight_bytes": meta.total_weight_bytes,
            "dominant_quant_type": meta.dominant_quant_type,
            "assignment_complete": assignment.is_complete,
            "unassigned_layers": assignment.unassigned_layers,
            "pipeline_order":   assignment.pipeline_order,
            "assignment":       {k: v for k, v in assignment.assignment.items() if v},
        }
    except FileNotFoundError as e:
        STATE.status = "error"; STATE.error = str(e)
        raise HTTPException(404, str(e))
    except Exception as e:
        STATE.status = "error"; STATE.error = str(e)
        log.exception("load-model failed")
        raise HTTPException(500, str(e))