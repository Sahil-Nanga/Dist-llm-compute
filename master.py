from __future__ import annotations

import asyncio
import gc
import json
import logging
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    CMD_ACK, CMD_ASSIGN_LAYERS, CMD_HELLO, CMD_RAM_REPORT,
    CMD_RESET, CMD_SET_NEXT_HOP, CMD_START_INFERENCE,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P,
    MASTER_API_HOST, MASTER_API_PORT, MiB, RAM_SAFETY_MARGIN,
    ZMQ_COLLECT_PORT, ZMQ_CONTROL_PORT, ZMQ_PIPELINE_PORT, ZMQ_WEIGHT_PORT,
)
from gguf_loader import (
    build_index, load_embedding_weights, load_layer_streaming,
    load_layers_range, load_output_weights,
)
from inference_engine import EmbeddingLayer, LayerRangeEngine, OutputLayer, sample_token
from networking import (
    ControlChannel, DeviceDiscovery, PipelineChannel, TensorTransport,
    WeightChannel, WorkerServiceInfo, ZMQContext, get_local_ip,
)
from profiler import (
    DeviceProfile, GGUFMetadata, LayerAssignment,
    compute_layer_assignment, get_local_device_profile,
    parse_model, ram_bytes_from_report,
)
from tokenizer import GGUFTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(name)s - %(message)s",
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

    def to_status_dict(self) -> dict:
        return {
            "status":         self.status,
            "error":          self.error,
            "model_loaded":   self.model_path is not None,
            "model_path":     self.model_path,
            "n_workers":      len(self.workers),
            "pipeline_ready": len(self.pipeline_order) > 0,
            "master_ip":      get_local_ip(),
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
    return HTMLResponse("<h1>Dashboard not found - place frontend/ next to master.py</h1>")


class LoadModelRequest(BaseModel):
    model_path: str


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K


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
        assignment = STATE.assignment.assignment.get(dev_id, []) if STATE.assignment else []
        workers.append({
            "device_id":           dev_id,
            "host":                info.host,
            "port":                info.port,
            "total_ram_bytes":     profile.total_ram_bytes if profile else None,
            "available_ram_bytes": profile.available_ram_bytes if profile else None,
            "usable_ram_bytes":    profile.usable_ram_bytes if profile else None,
            "assigned_layers":     assignment,
            "n_layers":            len(assignment),
        })

    master_profile = get_local_device_profile("master")
    master_assignment = STATE.assignment.assignment.get("master", []) if STATE.assignment else []
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
    return {
        "models":     [str(f) for f in sorted(p.glob("**/*.gguf"))],
        "models_dir": str(p.resolve()),
    }


@app.post("/api/load-model")
async def api_load_model(req: LoadModelRequest):
    STATE.status = "loading_model"
    STATE.error = None
    try:
        meta = parse_model(req.model_path)
        STATE.model_metadata = meta
        STATE.model_path = req.model_path

        profiles: Dict[str, DeviceProfile] = {"master": get_local_device_profile("master")}
        profiles.update(STATE.worker_profiles)

        assignment = compute_layer_assignment(meta, profiles)
        STATE.assignment = assignment
        STATE.pipeline_order = assignment.pipeline_order

        tok_info = {}
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
            log.warning("Tokenizer loading failed: %s", tok_err)
            tok_info = {"tokenizer_type": "byte_fallback", "error": str(tok_err)}

        STATE.status = "model_ready"
        return {
            "architecture":        meta.architecture,
            "n_layers":            meta.n_layers,
            "embedding_dim":       meta.embedding_dim,
            "n_heads":             meta.n_heads,
            "n_kv_heads":          meta.n_kv_heads,
            "intermediate_dim":    meta.intermediate_dim,
            "vocab_size":          meta.vocab_size,
            "bytes_per_layer":     meta.bytes_per_layer,
            "total_weight_bytes":  meta.total_weight_bytes,
            "dominant_quant_type": meta.dominant_quant_type,
            "assignment_complete": assignment.is_complete,
            "unassigned_layers":   assignment.unassigned_layers,
            "pipeline_order":      assignment.pipeline_order,
            "assignment":          {k: v for k, v in assignment.assignment.items() if v},
            "tokenizer":           tok_info,
        }
    except FileNotFoundError as e:
        STATE.status = "error"; STATE.error = str(e)
        raise HTTPException(404, str(e))
    except Exception as e:
        STATE.status = "error"; STATE.error = str(e)
        log.exception("load-model failed")
        raise HTTPException(500, str(e))


@app.get("/api/layer-assignment")
async def api_layer_assignment():
    if STATE.assignment is None:
        raise HTTPException(404, "No layer assignment yet. POST /api/load-model first.")
    meta = STATE.assignment.model_metadata
    rows = []
    for dev in STATE.assignment.pipeline_order:
        layers = STATE.assignment.assignment[dev]
        rows.append({
            "device_id":      dev,
            "layer_start":    layers[0] if layers else None,
            "layer_end":      layers[-1] if layers else None,
            "n_layers":       len(layers),
            "layer_indices":  layers,
            "mem_used_bytes": len(layers) * meta.bytes_per_layer,
        })
    return {
        "total_layers":      meta.n_layers,
        "bytes_per_layer":   meta.bytes_per_layer,
        "unassigned_layers": STATE.assignment.unassigned_layers,
        "pipeline":          rows,
    }


@app.get("/api/tokenizer")
async def api_tokenizer():
    if STATE.tokenizer is None:
        raise HTTPException(404, "No tokenizer loaded yet.")
    tok = STATE.tokenizer
    return {
        "model_type":        tok.model_type,
        "vocab_size":        tok.vocab_size,
        "bos_id":            tok.bos_id,
        "eos_id":            tok.eos_id,
        "has_chat_template": tok.has_chat_template(),
        "special_tokens":    tok.get_special_tokens(),
    }


@app.post("/api/tokenizer/encode")
async def api_encode(body: dict):
    if STATE.tokenizer is None:
        raise HTTPException(404, "No tokenizer loaded")
    text = body.get("text", "")
    add_bos = body.get("add_bos", True)
    if body.get("use_chat_template", False):
        text = STATE.tokenizer.format_chat(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
        )
        add_bos = False
    ids = STATE.tokenizer.encode(text, add_bos=add_bos)
    return {"token_ids": ids, "n_tokens": len(ids), "formatted_text": text}


@app.post("/api/tokenizer/decode")
async def api_decode(body: dict):
    if STATE.tokenizer is None:
        raise HTTPException(404, "No tokenizer loaded")
    return {"text": STATE.tokenizer.decode(body.get("token_ids", []))}


@app.post("/api/deploy")
async def api_deploy(background: BackgroundTasks):
    if STATE.model_path is None or STATE.assignment is None:
        raise HTTPException(400, "No model loaded. POST /api/load-model first.")
    STATE.status = "deploying"
    background.add_task(_deploy_task)
    return {"message": "Deployment started"}


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    if STATE.status not in ("deployed", "ready"):
        raise HTTPException(400, f"System not ready (status={STATE.status}).")
    return StreamingResponse(
        _generate_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
async def api_reset():
    for dev_id, ctrl in STATE.worker_ctrl.items():
        try:
            ctrl.send_command(CMD_RESET)
            log.info("Reset worker %s", dev_id)
        except Exception as e:
            log.warning("Could not reset worker %s: %s", dev_id, e)
    if STATE.master_engine:
        for rid in list(STATE.master_engine.kv_caches.keys()):
            STATE.master_engine.clear_cache(rid)
    return {"message": "Reset sent to all nodes"}


async def _discovery_task(timeout: float):
    loop = asyncio.get_event_loop()

    def _discover():
        discovery = DeviceDiscovery()
        found = discovery.discover(timeout=timeout)
        discovery.close()
        return found

    found: List[WorkerServiceInfo] = await loop.run_in_executor(None, _discover)
    zmq = STATE.zmq or ZMQContext()
    STATE.zmq = zmq

    for info in found:
        dev_id = info.device_id
        if dev_id in STATE.workers:
            continue
        STATE.workers[dev_id] = info

        ctrl = ControlChannel(zmq, is_server=False)
        ctrl.connect_client(info.host, ZMQ_CONTROL_PORT)
        try:
            ctrl.send_command(CMD_HELLO)
        except Exception as e:
            log.warning("Handshake failed for %s: %s", dev_id, e)

        try:
            r_cmd, r_data = ctrl.send_command(CMD_RAM_REPORT)
            if r_cmd == CMD_ACK and r_data:
                STATE.worker_profiles[dev_id] = ram_bytes_from_report(r_data)
                log.info("Worker %s: %.1f GiB available",
                         dev_id, STATE.worker_profiles[dev_id].available_mib / 1024)
        except Exception as e:
            log.warning("RAM query failed for %s: %s", dev_id, e)

        STATE.worker_ctrl[dev_id] = ctrl
        wch = WeightChannel(zmq)
        wch.connect_sender(info.host, ZMQ_WEIGHT_PORT)
        STATE.worker_weight[dev_id] = wch

    STATE.status = "discovered" if STATE.workers else "no_workers"
    log.info("Discovery complete. Workers: %s", list(STATE.workers.keys()))


async def _deploy_task():
    assignment = STATE.assignment
    meta = STATE.model_metadata
    path = STATE.model_path

    if not assignment or not meta or not path:
        STATE.status = "error"
        STATE.error = "Missing assignment/model before deploy"
        return

    try:
        zmq = STATE.zmq or ZMQContext()
        STATE.zmq = zmq
        loop = asyncio.get_event_loop()

        log.info("Building GGUF tensor index...")
        idx = await loop.run_in_executor(None, build_index, path)
        log.info("Index ready: %d tensors", len(idx.tensors))

        for dev_id in assignment.pipeline_order:
            if dev_id == "master":
                continue
            layers = assignment.assignment.get(dev_id, [])
            if not layers:
                continue
            ctrl = STATE.worker_ctrl.get(dev_id)
            wch = STATE.worker_weight.get(dev_id)
            if ctrl is None or wch is None:
                log.error("No ZMQ connection to %s - skipping", dev_id)
                continue

            ctrl.send_command(CMD_ASSIGN_LAYERS, {
                "layer_indices": layers,
                "n_heads":       meta.n_heads,
                "n_kv_heads":    meta.n_kv_heads,
                "rope_freq_base": meta.rope_freq_base,
            })
            log.info("Streaming %d layers to %s...", len(layers), dev_id)

            async def _pipeline_send(layers, wch, dev_id):
                n = len(layers)
                prefetch_future = loop.run_in_executor(
                    None, load_layer_streaming, idx, layers[0]
                )
                for i, layer_idx in enumerate(layers):
                    weights = await prefetch_future
                    if i + 1 < n:
                        prefetch_future = loop.run_in_executor(
                            None, load_layer_streaming, idx, layers[i + 1]
                        )
                    mem_mib = sum(a.nbytes for k, a in weights.items() if not k.startswith("__")) / (1024**2)
                    log.info("  layer %d/%d -> %s (%.0f MiB)", i+1, n, dev_id, mem_mib)
                    await loop.run_in_executor(None, wch.send_layers, [layer_idx], [weights])
                    del weights
                    gc.collect()

            await _pipeline_send(layers, wch, dev_id)
            log.info("All layers sent to %s", dev_id)

        master_layers = assignment.assignment.get("master", [])
        if master_layers:
            log.info("Loading %d master layers...", len(master_layers))
            master_weights_list = []
            for i, layer_idx in enumerate(master_layers):
                w = await loop.run_in_executor(None, load_layer_streaming, idx, layer_idx)
                master_weights_list.append(w)
                used = sum(a.nbytes for k, a in w.items() if not k.startswith("__")) / (1024 ** 2)
                log.info("  master layer %d/%d loaded (%.0f MiB)", i + 1, len(master_layers), used)
                gc.collect()
            STATE.master_engine = LayerRangeEngine(
                layer_indices=master_layers,
                weights_list=master_weights_list,
                n_heads=meta.n_heads,
                n_kv_heads=meta.n_kv_heads,
                rope_freq_base=meta.rope_freq_base,
            )
            log.info("Master engine ready: layers %d-%d", master_layers[0], master_layers[-1])

        log.info("Loading embedding weights...")
        embed_w = await loop.run_in_executor(None, load_embedding_weights, path)
        STATE.embed_layer = EmbeddingLayer(embed_w)
        del embed_w
        gc.collect()

        log.info("Loading output weights...")
        output_w = await loop.run_in_executor(None, load_output_weights, path)
        STATE.output_layer = OutputLayer(output_w)
        del output_w
        gc.collect()

        _setup_pipeline(zmq)

        for dev_id in assignment.pipeline_order:
            if dev_id == "master":
                continue
            ctrl = STATE.worker_ctrl.get(dev_id)
            if ctrl:
                ctrl.send_command(CMD_START_INFERENCE)

        STATE.status = "deployed"
        log.info("Deployment complete. Pipeline: %s", assignment.pipeline_order)

    except Exception as e:
        STATE.status = "error"
        STATE.error = str(e)
        log.exception("Deployment failed")


def _setup_pipeline(zmq: ZMQContext):
    order = STATE.pipeline_order
    my_ip = get_local_ip()
    has_workers = any(d != "master" for d in order)

    if not has_workers:
        log.info("All layers on master - running fully local inference")
        return

    first_worker = next(d for d in order if d != "master")
    first_info = STATE.workers[first_worker]

    STATE.pipe_collect = PipelineChannel(zmq)
    STATE.pipe_collect.bind_input(ZMQ_COLLECT_PORT)

    STATE.pipe_out = PipelineChannel(zmq)
    STATE.pipe_out.connect_output(first_info.host, ZMQ_PIPELINE_PORT)

    for i, dev_id in enumerate(order):
        if dev_id == "master":
            continue
        ctrl = STATE.worker_ctrl[dev_id]
        is_last = (i == len(order) - 1) or all(d == "master" for d in order[i + 1:])

        if is_last:
            ctrl.send_command(CMD_SET_NEXT_HOP, {
                "is_last":      True,
                "host":         my_ip,
                "collect_port": ZMQ_COLLECT_PORT,
            })
        else:
            next_dev = next(d for d in order[i + 1:] if d != "master")
            next_info = STATE.workers[next_dev]
            ctrl.send_command(CMD_SET_NEXT_HOP, {
                "is_last": False,
                "host":    next_info.host,
                "port":    ZMQ_PIPELINE_PORT,
            })

    log.info("Pipeline wired: %s", " -> ".join(order))


async def _generate_stream(req: GenerateRequest):
    if STATE.embed_layer is None or STATE.output_layer is None:
        yield _sse_event({"error": "Model not deployed"})
        return

    if STATE.tokenizer is None:
        yield _sse_event({"error": "Tokenizer not loaded"})
        return

    tok = STATE.tokenizer
    order = STATE.pipeline_order
    request_id = int(time.time() * 1000) % (2 ** 32)

    formatted_prompt = tok.format_chat(
        [{"role": "user", "content": req.prompt}],
        add_generation_prompt=True,
    )
    token_ids = tok.encode(formatted_prompt, add_bos=False)
    token_ids_np = np.array([token_ids], dtype=np.int64)

    generated_tokens = []
    eos_token = tok.eos_id
    loop = asyncio.get_event_loop()

    def _run_inference():
        nonlocal token_ids_np
        positions = np.arange(token_ids_np.shape[1], dtype=np.int64)
        hidden = STATE.embed_layer.forward(token_ids_np)
        reset = True

        for _ in range(req.max_new_tokens):
            meta_payload = {
                "positions":   positions.tolist(),
                "reset_cache": reset,
                "request_id":  request_id,
            }

            if STATE.master_engine and "master" in order:
                hidden = STATE.master_engine.forward(
                    hidden, positions,
                    request_id=request_id, reset_cache=reset,
                )

            if STATE.pipe_out is not None:
                STATE.pipe_out.send(request_id, hidden, meta_payload)
                _, hidden, _ = STATE.pipe_collect.recv(timeout_ms=120_000)

            logits = STATE.output_layer.forward(hidden)
            next_id = sample_token(
                logits[0, -1],
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )

            generated_tokens.append(next_id)
            yield {"token_id": next_id, "token_text": tok.decode_token(next_id)}

            if next_id == eos_token or next_id == 128009:
                break

            token_ids_np = np.array([[next_id]], dtype=np.int64)
            positions = np.array([positions[-1] + 1], dtype=np.int64)
            hidden = STATE.embed_layer.forward(token_ids_np)
            reset = False

    gen_queue: asyncio.Queue = asyncio.Queue()

    def _worker_thread():
        try:
            for event in _run_inference():
                asyncio.run_coroutine_threadsafe(gen_queue.put(event), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(gen_queue.put({"error": str(e)}), loop)
        asyncio.run_coroutine_threadsafe(gen_queue.put(None), loop)

    threading.Thread(target=_worker_thread, daemon=True).start()

    while True:
        event = await gen_queue.get()
        if event is None:
            yield _sse_event({"done": True, "total_tokens": len(generated_tokens)})
            break
        yield _sse_event(event)


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Distributed LLM - Master Node")
    p.add_argument("--models-dir", default="./models")
    p.add_argument("--port", type=int, default=MASTER_API_PORT)
    p.add_argument("--host", default=MASTER_API_HOST)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--auto-discover", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    log.info("Master Node | http://%s:%d", get_local_ip(), args.port)

    if args.auto_discover:
        async def _startup():
            asyncio.create_task(_discovery_task(15.0))
        app.add_event_handler("startup", _startup)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()