from __future__ import annotations

import io
import json
import queue
import socket
import struct
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq
from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf

from config import (
    ZEROCONF_SERVICE_TYPE,
    ZEROCONF_DISCOVERY_TIMEOUT,
    ZMQ_CONTROL_PORT,
    ZMQ_WEIGHT_PORT,
    ZMQ_PIPELINE_PORT,
    ZMQ_COLLECT_PORT,
    CMD_ACK,
    CMD_ERROR,
)

log = logging.getLogger(__name__)



def get_local_ip() -> str:
    """Return the primary LAN IP of this machine (not 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class TensorTransport:
    """
    Lightweight numpy-array framing.

    Wire format (bytes):
      [4 bytes]  dtype string length  (uint32 LE)
      [N bytes]  dtype string         (UTF-8)
      [4 bytes]  number of dimensions (uint32 LE)
      [8*ndim]   shape dims           (uint64 LE each)
      [rest]     raw C-contiguous data

    This format is ~3× faster than pickle for large float32 arrays
    and avoids arbitrary code execution on deserialisation.
    """

    @staticmethod
    def pack(arr: np.ndarray) -> bytes:
        arr   = np.ascontiguousarray(arr)
        dtype = arr.dtype.str.encode()        
        ndim  = arr.ndim
        header = struct.pack("<I", len(dtype)) + dtype
        header += struct.pack("<I", ndim)
        header += struct.pack(f"<{ndim}Q", *arr.shape)
        return header + arr.tobytes()

    @staticmethod
    def unpack(data: bytes) -> np.ndarray:
        offset = 0
        dtype_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
        dtype_str = data[offset:offset + dtype_len].decode(); offset += dtype_len
        ndim      = struct.unpack_from("<I", data, offset)[0]; offset += 4
        shape     = struct.unpack_from(f"<{ndim}Q", data, offset); offset += 8 * ndim
        arr = np.frombuffer(data[offset:], dtype=np.dtype(dtype_str)).reshape(shape)
        return arr.copy()  

    @staticmethod
    def pack_dict(d: Dict[str, np.ndarray]) -> bytes:
        """Serialise a {name: tensor} dict (used for layer weights)."""
        buf = io.BytesIO()
        n = len(d)
        buf.write(struct.pack("<I", n))
        for name, arr in d.items():
            name_b = name.encode()
            buf.write(struct.pack("<I", len(name_b)))
            buf.write(name_b)
            t = TensorTransport.pack(arr)
            buf.write(struct.pack("<Q", len(t)))
            buf.write(t)
        return buf.getvalue()

    @staticmethod
    def unpack_dict(data: bytes) -> Dict[str, np.ndarray]:
        offset = 0
        n = struct.unpack_from("<I", data, offset)[0]; offset += 4
        result: Dict[str, np.ndarray] = {}
        for _ in range(n):
            nlen = struct.unpack_from("<I", data, offset)[0]; offset += 4
            name = data[offset:offset + nlen].decode(); offset += nlen
            tlen = struct.unpack_from("<Q", data, offset)[0]; offset += 8
            arr  = TensorTransport.unpack(data[offset:offset + tlen]); offset += tlen
            result[name] = arr
        return result



class ZMQContext:
    """
    Singleton-style ZMQ context wrapper.
    All sockets created here are tracked and closed via shutdown().
    """

    def __init__(self, io_threads: int = 4):
        self._ctx = zmq.Context(io_threads)
        self._sockets: List[zmq.Socket] = []
        self._lock = threading.Lock()


    def _make(self, socket_type: int, **opts) -> zmq.Socket:
        s = self._ctx.socket(socket_type)
        for k, v in opts.items():
            s.setsockopt_string(getattr(zmq, k.upper()), v) if isinstance(v, str) \
                else s.setsockopt(getattr(zmq, k.upper()), v)
        with self._lock:
            self._sockets.append(s)
        return s

    def req_socket(self, *, linger: int = 1000) -> zmq.Socket:
        return self._make(zmq.REQ, linger=linger, rcvtimeo=30_000, sndtimeo=30_000)

    def rep_socket(self, *, linger: int = 1000) -> zmq.Socket:
        return self._make(zmq.REP, linger=linger)

    def push_socket(self, *, linger: int = 5000, hwm: int = 16) -> zmq.Socket:
        s = self._make(zmq.PUSH, linger=linger)
        s.setsockopt(zmq.SNDHWM, hwm)
        return s

    def pull_socket(self, *, linger: int = 5000, hwm: int = 16) -> zmq.Socket:
        s = self._make(zmq.PULL, linger=linger)
        s.setsockopt(zmq.RCVHWM, hwm)
        return s

    def router_socket(self, *, linger: int = 1000) -> zmq.Socket:
        return self._make(zmq.ROUTER, linger=linger, router_mandatory=1)

    def dealer_socket(self, *, linger: int = 1000) -> zmq.Socket:
        return self._make(zmq.DEALER, linger=linger)

    def shutdown(self):
        log.info("ZMQContext: closing %d socket(s)", len(self._sockets))
        for s in self._sockets:
            try:
                s.close()
            except Exception:
                pass
        self._ctx.term()



class ControlChannel:
    """
    Synchronous REQ/REP command channel.

    Master side:   connect REQ to each worker's bound REP.
    Worker side:   bind REP, call recv_command() in a loop.
    """

    ENCODING = "utf-8"

    def __init__(self, zmq_ctx: ZMQContext, is_server: bool):
        self._ctx       = zmq_ctx
        self._is_server = is_server
        self._sock: Optional[zmq.Socket] = None

    def bind_server(self, port: int = ZMQ_CONTROL_PORT):
        self._sock = self._ctx.rep_socket()
        addr = f"tcp://*:{port}"
        self._sock.bind(addr)
        log.info("Control channel bound on %s", addr)

    def connect_client(self, host: str, port: int = ZMQ_CONTROL_PORT):
        self._sock = self._ctx.req_socket()
        addr = f"tcp://{host}:{port}"
        self._sock.connect(addr)
        log.info("Control channel connected to %s", addr)


    def recv_command(self) -> Tuple[bytes, Optional[dict]]:
        """Blocking. Returns (command_bytes, payload_dict or None)."""
        frames = self._sock.recv_multipart()
        cmd    = frames[0]
        payload = json.loads(frames[1]) if len(frames) > 1 else None
        return cmd, payload

    def send_ack(self, data: Optional[dict] = None):
        frames = [CMD_ACK]
        if data:
            frames.append(json.dumps(data).encode())
        self._sock.send_multipart(frames)

    def send_error(self, msg: str):
        self._sock.send_multipart([CMD_ERROR, msg.encode()])


    def send_command(self, cmd: bytes, payload: Optional[dict] = None) -> Tuple[bytes, Optional[dict]]:
        """Send a command, block for ACK. Returns (response_cmd, response_data)."""
        frames = [cmd]
        if payload:
            frames.append(json.dumps(payload).encode())
        self._sock.send_multipart(frames)
        resp = self._sock.recv_multipart()
        r_cmd     = resp[0]
        r_payload = json.loads(resp[1]) if len(resp) > 1 else None
        return r_cmd, r_payload



class PipelineChannel:
    """
    PUSH/PULL channel for passing hidden-state tensors along the inference pipeline.

    Each device:
      • binds a PULL socket on its pipeline_in_port
      • connects a PUSH socket to the NEXT device's pipeline_in_port

    Data format on wire:
      frame[0] = 8-byte request ID (uint64 LE)
      frame[1] = packed tensor (TensorTransport.pack)
      frame[2] = JSON metadata (token positions, etc.)
    """

    def __init__(self, zmq_ctx: ZMQContext):
        self._ctx  = zmq_ctx
        self._pull: Optional[zmq.Socket] = None
        self._push: Optional[zmq.Socket] = None

    def bind_input(self, port: int = ZMQ_PIPELINE_PORT):
        self._pull = self._ctx.pull_socket()
        self._pull.bind(f"tcp://*:{port}")
        log.info("Pipeline INPUT bound on port %d", port)

    def connect_output(self, host: str, port: int = ZMQ_PIPELINE_PORT):
        self._push = self._ctx.push_socket()
        self._push.connect(f"tcp://{host}:{port}")
        log.info("Pipeline OUTPUT connected to %s:%d", host, port)

    def send(self, request_id: int, tensor: np.ndarray, meta: dict):
        rid    = struct.pack("<Q", request_id)
        tdata  = TensorTransport.pack(tensor)
        mdata  = json.dumps(meta).encode()
        self._push.send_multipart([rid, tdata, mdata])

    def recv(self, timeout_ms: int = 60_000) -> Tuple[int, np.ndarray, dict]:
        if self._pull.poll(timeout_ms) == 0:
            raise TimeoutError("Pipeline recv timed out")
        frames    = self._pull.recv_multipart()
        request_id = struct.unpack("<Q", frames[0])[0]
        tensor     = TensorTransport.unpack(frames[1])
        meta       = json.loads(frames[2])
        return request_id, tensor, meta



class WeightChannel:
    """
    PUSH/PULL channel for master → worker layer weight distribution.

    Because weight tensors can be hundreds of MB, we use a dedicated
    channel with high-watermark = 1 to avoid memory blow-up.

    Wire format (single message):
      frame[0] = JSON header: {layer_indices, n_chunks}
      frame[1..n] = serialised weight dicts (one per layer, via TensorTransport.pack_dict)
    """

    def __init__(self, zmq_ctx: ZMQContext):
        self._ctx  = zmq_ctx
        self._pull: Optional[zmq.Socket] = None
        self._push: Optional[zmq.Socket] = None

    def bind_receiver(self, port: int = ZMQ_WEIGHT_PORT):
        self._pull = self._ctx.pull_socket(hwm=2)
        self._pull.bind(f"tcp://*:{port}")
        log.info("Weight receiver bound on port %d", port)

    def connect_sender(self, host: str, port: int = ZMQ_WEIGHT_PORT):
        self._push = self._ctx.push_socket(hwm=2)
        self._push.connect(f"tcp://{host}:{port}")
        log.info("Weight sender connected to %s:%d", host, port)

    def send_layers(self, layer_indices: List[int], weights: List[Dict[str, np.ndarray]]):
        """Send a list of layer weight dicts to a worker."""
        header = json.dumps({"layer_indices": layer_indices}).encode()
        frames = [header] + [TensorTransport.pack_dict(w) for w in weights]
        self._push.send_multipart(frames)
        log.info("Sent %d layer weight packages", len(weights))

    def recv_layers(self, timeout_ms: int = 300_000) -> Tuple[List[int], List[Dict[str, np.ndarray]]]:
        """Blocking. Returns (layer_indices, [weight_dict, ...])."""
        if self._pull.poll(timeout_ms) == 0:
            raise TimeoutError("Weight recv timed out waiting for master")
        frames  = self._pull.recv_multipart()
        header  = json.loads(frames[0])
        indices = header["layer_indices"]
        weights = [TensorTransport.unpack_dict(f) for f in frames[1:]]
        log.info("Received weights for layers: %s", indices)
        return indices, weights



@dataclass
class WorkerServiceInfo:
    name: str
    host: str
    port: int
    device_id: str
    properties: Dict[str, str]


class WorkerListener(ServiceListener):
    """
    Zeroconf callback listener.
    Discovered worker services are pushed into self.queue.
    """

    def __init__(self):
        self.queue: queue.Queue[WorkerServiceInfo] = queue.Queue()
        self.removed: queue.Queue[str] = queue.Queue()

    def add_service(self, zc: Zeroconf, service_type: str, name: str):
        info = zc.get_service_info(service_type, name)
        if info is None:
            return
        host  = socket.inet_ntoa(info.addresses[0]) if info.addresses else "127.0.0.1"
        props = {k.decode(): v.decode() if isinstance(v, bytes) else v
                 for k, v in (info.properties or {}).items()}
        w = WorkerServiceInfo(
            name=name,
            host=host,
            port=info.port,
            device_id=props.get("device_id", name),
            properties=props,
        )
        log.info("Discovered worker: %s @ %s:%d", w.device_id, w.host, w.port)
        self.queue.put(w)

    def remove_service(self, zc: Zeroconf, service_type: str, name: str):
        log.info("Worker left: %s", name)
        self.removed.put(name)

    def update_service(self, zc: Zeroconf, service_type: str, name: str):
        self.add_service(zc, service_type, name)


class DeviceDiscovery:
    """
    Zeroconf-based mDNS discovery for the master node.

    Usage:
        discovery = DeviceDiscovery()
        workers = discovery.discover(timeout=15.0)

    Each item in workers is a WorkerServiceInfo instance.
    """

    def __init__(self):
        self._zc = Zeroconf()
        self._listener = WorkerListener()

    def start_browsing(self):
        self._browser = ServiceBrowser(self._zc, ZEROCONF_SERVICE_TYPE, self._listener)
        log.info("Started mDNS browsing for %s", ZEROCONF_SERVICE_TYPE)

    def discover(self, timeout: float = ZEROCONF_DISCOVERY_TIMEOUT) -> List[WorkerServiceInfo]:
        self.start_browsing()
        deadline = time.time() + timeout
        found: List[WorkerServiceInfo] = []
        while time.time() < deadline:
            try:
                info = self._listener.queue.get(timeout=0.5)
                found.append(info)
            except queue.Empty:
                pass
        log.info("Discovery complete. Found %d worker(s).", len(found))
        return found

    def close(self):
        self._zc.close()


class WorkerAnnouncer:
    """
    Zeroconf service registrar for worker nodes.

    Registers a _llmdist._tcp.local. service so the master can find us.
    Properties carry the device_id and port numbers the master will connect to.
    """

    def __init__(self, device_id: str, host: Optional[str] = None):
        self._device_id = device_id
        self._host      = host or get_local_ip()
        self._zc: Optional[Zeroconf] = None
        self._info: Optional[ServiceInfo] = None

    def announce(
        self,
        control_port: int = ZMQ_CONTROL_PORT,
        weight_port: int  = ZMQ_WEIGHT_PORT,
        pipeline_port: int = ZMQ_PIPELINE_PORT,
    ):
        self._zc = Zeroconf()
        service_name = f"{self._device_id}.{ZEROCONF_SERVICE_TYPE}"
        props = {
            "device_id":    self._device_id,
            "control_port": str(control_port),
            "weight_port":  str(weight_port),
            "pipeline_port": str(pipeline_port),
            "version":      "2",
        }
        self._info = ServiceInfo(
            type_=ZEROCONF_SERVICE_TYPE,
            name=service_name,
            addresses=[socket.inet_aton(self._host)],
            port=control_port,
            properties=props,
            server=f"{self._device_id}.local.",
        )
        self._zc.register_service(self._info)
        log.info("Announced service '%s' at %s:%d", service_name, self._host, control_port)

    def revoke(self):
        if self._zc and self._info:
            self._zc.unregister_service(self._info)
            self._zc.close()
            log.info("Service '%s' revoked", self._device_id)