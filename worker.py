from __future__ import annotations

import argparse
import json
import logging
import signal
import struct
import sys
import threading
import time
from typing import Optional

import numpy as np
import psutil

from config import (
    ZMQ_CONTROL_PORT, ZMQ_WEIGHT_PORT, ZMQ_PIPELINE_PORT,
    CMD_HELLO, CMD_RAM_REPORT, CMD_ASSIGN_LAYERS, CMD_WEIGHTS_READY,
    CMD_SET_NEXT_HOP, CMD_START_INFERENCE, CMD_RESET, CMD_ACK, CMD_ERROR,
    PROTOCOL_VERSION,
    RAM_SAFETY_MARGIN,
)
from networking import (
    ZMQContext, ControlChannel, WeightChannel, PipelineChannel,
    WorkerAnnouncer, get_local_ip, TensorTransport,
)
from inference_engine import LayerRangeEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")



class WorkerNode:
    """
    Full worker-node state machine.

    States: IDLE → CONFIGURED → READY → INFERRING → IDLE (loop)
    """

    def __init__(self, device_id: str, host: str):
        self.device_id   = device_id
        self.host        = host
        self._running    = True
        self._engine: Optional[LayerRangeEngine] = None
        self._n_heads:    int = 32
        self._n_kv_heads: int = 32
        self._next_host: Optional[str] = None
        self._next_port: int = ZMQ_PIPELINE_PORT
        self._is_last_node: bool = False

        self._zmq   = ZMQContext()
        self._ctrl  = ControlChannel(self._zmq, is_server=True)
        self._weight_ch = WeightChannel(self._zmq)
        self._pipe_in   = PipelineChannel(self._zmq)
        self._pipe_out  = PipelineChannel(self._zmq)

        self._announcer = WorkerAnnouncer(device_id, host)


    def start(self):
        log.info("═" * 60)
        log.info("  Worker Node  |  device_id=%s  |  host=%s", self.device_id, self.host)
        log.info("═" * 60)

        self._ctrl.bind_server(ZMQ_CONTROL_PORT)
        self._weight_ch.bind_receiver(ZMQ_WEIGHT_PORT)
        self._pipe_in.bind_input(ZMQ_PIPELINE_PORT)

        self._announcer.announce(
            control_port=ZMQ_CONTROL_PORT,
            weight_port=ZMQ_WEIGHT_PORT,
            pipeline_port=ZMQ_PIPELINE_PORT,
        )

        try:
            self._command_loop()
        except KeyboardInterrupt:
            log.info("Keyboard interrupt — shutting down")
        finally:
            self._shutdown()


    def _command_loop(self):
        log.info("Waiting for commands from master …")
        while self._running:
            try:
                cmd, payload = self._ctrl.recv_command()
            except Exception as exc:
                log.warning("Control recv error: %s", exc)
                continue

            log.debug("CMD: %s  payload=%s", cmd, payload)

            if cmd == CMD_HELLO:
                self._handle_hello(payload)

            elif cmd == CMD_RAM_REPORT:
                self._handle_ram_report()

            elif cmd == CMD_ASSIGN_LAYERS:
                self._handle_assign_layers(payload)

            elif cmd == CMD_SET_NEXT_HOP:
                self._handle_set_next_hop(payload)

            elif cmd == CMD_START_INFERENCE:
                self._ctrl.send_ack({"status": "running"})
                self._inference_loop()

            elif cmd == CMD_RESET:
                self._handle_reset()
                self._ctrl.send_ack()

            else:
                log.warning("Unknown command: %s", cmd)
                self._ctrl.send_error(f"Unknown command: {cmd!r}")


    def _handle_hello(self, payload: Optional[dict]):
        resp = {
            "device_id":       self.device_id,
            "host":            self.host,
            "protocol_version": PROTOCOL_VERSION,
        }
        self._ctrl.send_ack(resp)
        log.info("Handshake complete with master")

    def _handle_ram_report(self):
        mem = psutil.virtual_memory()
        report = {
            "device_id":          self.device_id,
            "total_ram_bytes":    mem.total,
            "available_ram_bytes": mem.available,
            "usable_ram_bytes":   int(mem.available * RAM_SAFETY_MARGIN),
            "cpu_count":          psutil.cpu_count(),
            "cpu_percent":        psutil.cpu_percent(interval=0.1),
        }
        self._ctrl.send_ack(report)
        log.info(
            "RAM report sent: total=%.1f GiB  available=%.1f GiB",
            mem.total / (1024 ** 3),
            mem.available / (1024 ** 3),
        )

    def _handle_assign_layers(self, payload: dict):
        layer_indices   = payload["layer_indices"]
        self._n_heads    = payload.get("n_heads",    32)
        self._n_kv_heads = payload.get("n_kv_heads", 32)
        rope_freq_base   = payload.get("rope_freq_base", 10000.0)

        self._ctrl.send_ack({
            "status": "ready_for_weights",
            "layer_indices": layer_indices,
        })
        log.info("Accepted layer assignment: %s (n_heads=%d)", layer_indices, self._n_heads)

        log.info("Waiting for weight tensors …")
        recv_indices, weights_list = self._weight_ch.recv_layers(timeout_ms=600_000)

        assert recv_indices == layer_indices, (
            f"Layer index mismatch: expected {layer_indices}, got {recv_indices}"
        )

        self._engine = LayerRangeEngine(
            layer_indices=recv_indices,
            weights_list=weights_list,
            n_heads=self._n_heads,
            n_kv_heads=self._n_kv_heads,
            rope_freq_base=rope_freq_base,
        )
        log.info("✓ Weights loaded. Engine ready for layers %s–%s.",
                 recv_indices[0], recv_indices[-1])

    def _handle_set_next_hop(self, payload: dict):
        is_last         = payload.get("is_last", False)
        self._is_last_node = is_last

        if not is_last:
            self._next_host = payload["host"]
            self._next_port = int(payload.get("port", ZMQ_PIPELINE_PORT))
            self._pipe_out.connect_output(self._next_host, self._next_port)
            log.info("Next hop set: %s:%d", self._next_host, self._next_port)
        else:
            collect_host = payload["host"]
            collect_port = int(payload.get("collect_port", ZMQ_PIPELINE_PORT + 1))
            self._pipe_out.connect_output(collect_host, collect_port)
            log.info("This is the LAST node. Collector: %s:%d", collect_host, collect_port)

        self._ctrl.send_ack({"status": "hop_set"})

    def _handle_reset(self):
        if self._engine:
            for rid in list(self._engine.kv_caches.keys()):
                self._engine.clear_cache(rid)
        log.info("State reset")


    def _inference_loop(self):
        """
        Blocking loop: receive a hidden-state tensor, run forward pass, send onward.
        Runs in the main thread after CMD_START_INFERENCE is received.
        """
        if self._engine is None:
            log.error("Inference loop started but engine is not ready!")
            return

        log.info("▶ Entering inference loop (layers %s–%s)",
                 self._engine.layer_indices[0], self._engine.layer_indices[-1])

        while self._running:
            try:
                request_id, hidden, meta = self._pipe_in.recv(timeout_ms=5_000)
            except TimeoutError:
                continue
            except Exception as exc:
                log.error("Pipeline recv error: %s", exc)
                continue

            reset_cache = meta.get("reset_cache", False)
            positions   = np.array(meta["positions"], dtype=np.int64)

            t0 = time.perf_counter()
            try:
                hidden_out = self._engine.forward(
                    hidden, positions, request_id=request_id, reset_cache=reset_cache
                )
            except Exception as exc:
                log.error("Forward pass failed for req %d: %s", request_id, exc, exc_info=True)
                continue

            dt = (time.perf_counter() - t0) * 1000
            log.info(
                "req=%d  layers=%d  seq=%d  %.1f ms",
                request_id, len(self._engine.layer_indices),
                hidden_out.shape[1], dt,
            )

            try:
                self._pipe_out.send(request_id, hidden_out, meta)
            except Exception as exc:
                log.error("Pipeline send failed for req %d: %s", request_id, exc)


    def _shutdown(self):
        log.info("Shutting down …")
        self._running = False
        try:
            self._announcer.revoke()
        except Exception:
            pass
        self._zmq.shutdown()
        log.info("Worker stopped.")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distributed LLM — Worker Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device-id", default=None,
                   help="Unique ID for this device (default: hostname)")
    p.add_argument("--host", default=None,
                   help="IP address to advertise (default: auto-detect LAN IP)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    import socket as _socket
    device_id = args.device_id or _socket.gethostname().replace(".", "-")
    host      = args.host      or get_local_ip()

    node = WorkerNode(device_id=device_id, host=host)

    def _sig_handler(sig, frame):
        log.info("Signal %d received — stopping", sig)
        node._running = False
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    node.start()


if __name__ == "__main__":
    main()