"""
Live finger-motion visualizer for the Rokoko glove.

Reads the RGMP v2 stream (same protocol as the SDK's rgmp_stream.py) and serves
a browser page that renders the hand in 3D, updated live over Server-Sent Events.

Usage:
  python glove_viz/server.py                       # stream at 127.0.0.1:12276, page at http://127.0.0.1:8000
  python glove_viz/server.py --rgmp-port 12276 --http-port 8000

Standard library only. Protocol helpers are imported from the Rokoko SDK
examples folder ($ROKOKO_SDK_HOME/examples/python/rgmp_stream.py).
"""

import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from recording import SessionRecorder

HERE = Path(__file__).resolve().parent

_sdk_home = os.environ.get("ROKOKO_SDK_HOME")
if _sdk_home:
    sys.path.insert(0, str(Path(_sdk_home) / "examples" / "python"))
try:
    from rgmp_stream import (
        DATA_FRAME_HEADER_FMT,
        DATA_FRAME_HEADER_SIZE,
        MSG_PREFIX_DATA,
        MSG_PREFIX_DEFINITION,
        MSG_PREFIX_DISCONNECT,
        OUTER_HEADER_FMT,
        OUTER_HEADER_SIZE,
        build_group_lookup,
        recv_exact,
    )
except ImportError:
    sys.exit(
        "Could not import rgmp_stream.py. Set ROKOKO_SDK_HOME to the Rokoko Device SDK "
        "folder (it should contain examples/python/rgmp_stream.py)."
    )


def stream_key(desc) -> str:
    """Compact key per stream, e.g. 'TRANSFORM@hub', 'CUSTOM@hub:relative_emf_residual'."""
    key = desc.measure_type
    if desc.reference_frame:
        key += "@" + desc.reference_frame
    if desc.custom_label:
        key += ":" + desc.custom_label
    return key


class GloveState:
    """Latest frame per group, shared between the RGMP reader and HTTP clients."""

    def __init__(self, recorder: SessionRecorder | None = None) -> None:
        self.cond = threading.Condition()
        self.seq = 0
        self.status = "starting"
        self.device_id: str | int | None = None
        self.groups: dict[str, dict] = {}
        self.recorder = recorder
        self._rate_count: dict[str, int] = {}
        self._rate_t0 = time.monotonic()
        self.rates: dict[str, float] = {}

    def set_status(self, status: str) -> None:
        with self.cond:
            if status != self.status:
                print(f"# {status}", file=sys.stderr, flush=True)
            self.status = status
            self.seq += 1
            self.cond.notify_all()

    def start_recording(self, label: object) -> dict:
        if self.recorder is None:
            raise RuntimeError("Recording is not configured")
        with self.cond:
            if not self.status.startswith("streaming"):
                raise RuntimeError("Connect a Rokoko device before recording")
            return self.recorder.start(label, self.device_id)

    def set_definition(self, device_id: str | int | None, group_names: list[str]) -> None:
        with self.cond:
            self.device_id = device_id
            self.groups = {name: {"t": None, "s": {}} for name in group_names}
            self._rate_count = {}
            self._rate_t0 = time.monotonic()
            self.rates = {}
            status = f"streaming device {device_id} ({len(group_names)} groups)"
            if status != self.status:
                print(f"# {status}", file=sys.stderr, flush=True)
            self.status = status
            self.seq += 1
            self.cond.notify_all()

    def reset_stream(self, status: str) -> None:
        with self.cond:
            if status != self.status:
                print(f"# {status}", file=sys.stderr, flush=True)
            self.status = status
            self.device_id = None
            self.groups = {}
            self._rate_count = {}
            self._rate_t0 = time.monotonic()
            self.rates = {}
            if self.recorder is not None:
                self.recorder.close()
            self.seq += 1
            self.cond.notify_all()

    def update_group(self, name: str, ts: int, streams: dict) -> None:
        with self.cond:
            self.groups[name] = {"t": ts, "s": streams}
            device_id = self.device_id
            self._rate_count[name] = self._rate_count.get(name, 0) + 1
            now = time.monotonic()
            if now - self._rate_t0 >= 1.0:
                dt = now - self._rate_t0
                self.rates = {k: round(v / dt, 1) for k, v in self._rate_count.items()}
                self._rate_count = {}
                self._rate_t0 = now
            self.seq += 1
            self.cond.notify_all()
        if self.recorder is not None:
            self.recorder.record_group(device_id, name, ts, streams)

    def snapshot_json(self) -> str:
        with self.cond:
            snapshot = {
                "status": self.status,
                "device_id": self.device_id,
                "rates": self.rates,
                "groups": self.groups,
            }
        if self.recorder is not None:
            snapshot["recording"] = self.recorder.snapshot()
        return json.dumps(snapshot, separators=(",", ":"), allow_nan=False)


def rgmp_reader(host: str, port: int, state: GloveState) -> None:
    """Connect to the RGMP server, decode frames into state; reconnect forever."""
    while True:
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.settimeout(None)
                state.reset_stream(f"connected to {host}:{port}, waiting for definition")
                lookup: dict = {}
                names: dict[int, str] = {}
                while True:
                    prefix, length = struct.unpack(
                        OUTER_HEADER_FMT, recv_exact(sock, OUTER_HEADER_SIZE)
                    )
                    payload = recv_exact(sock, length)

                    if prefix == MSG_PREFIX_DEFINITION:
                        definition = json.loads(payload.decode("utf-8"))
                        if not isinstance(definition, dict):
                            raise ValueError("RGMP definition must be a JSON object")
                        groups = definition.get("groups", [])
                        if not isinstance(groups, list):
                            raise ValueError("RGMP definition groups must be a list")
                        lookup = build_group_lookup(definition, None, None, None)
                        names = {}
                        for i, group in enumerate(groups):
                            if not isinstance(group, dict):
                                continue
                            name = group.get("name")
                            names[i] = name if isinstance(name, str) and name else f"group_{i}"
                        state.set_definition(definition.get("device_id"), list(names.values()))

                    elif prefix == MSG_PREFIX_DATA and lookup:
                        if len(payload) < DATA_FRAME_HEADER_SIZE:
                            continue
                        _dev, group_id, ts = struct.unpack_from(
                            DATA_FRAME_HEADER_FMT, payload, 0
                        )
                        descs = lookup.get(group_id)
                        if not descs:
                            continue
                        body = memoryview(payload)[DATA_FRAME_HEADER_SIZE:]
                        streams: dict[str, dict] = {}
                        for d in descs:
                            vals = struct.unpack_from(d.struct_fmt, body, d.offset)
                            vals = []
                            for value in struct.unpack_from(d.struct_fmt, body, d.offset):
                                if isinstance(value, float):
                                    if not math.isfinite(value):
                                        raise ValueError("RGMP frame contains a non-finite value")
                                    value = round(value, 5)
                                vals.append(value)
                            streams.setdefault(d.target_frame, {})[stream_key(d)] = vals
                        state.update_group(names.get(group_id, f"group_{group_id}"), ts, streams)

                    elif prefix == MSG_PREFIX_DISCONNECT:
                        lookup = {}
                        names = {}
                        state.reset_stream("device disconnected, waiting")
        except (OSError, ConnectionError, struct.error, ValueError, TypeError, KeyError, AttributeError) as exc:
            state.reset_stream(f"waiting for RGMP stream on {host}:{port} ({exc})")
            time.sleep(1.0)


PAGES = {
    "/": "index.html",
    "/index.html": "index.html",
}


def make_handler(state: GloveState, max_fps: float):
    min_interval = 1.0 / max_fps
    recorder = state.recorder

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the console quiet
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            page = PAGES.get(path)
            if page:
                body = (HERE / page).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/recording":
                self._send_json(200, {"recording": recorder.snapshot() if recorder else None})
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with state.cond:
                            state.cond.wait_for(lambda: state.seq != last, timeout=1.0)
                            last = state.seq
                            data = state.snapshot_json()
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(min_interval)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if recorder is None or path not in ("/api/recording/start", "/api/recording/stop"):
                self.send_error(404)
                return

            if path == "/api/recording/start":
                body = self._read_json_object()
                if body is None:
                    return
                try:
                    result = state.start_recording(body.get("label"))
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self._send_json(409, {"error": str(exc)})
                except OSError as exc:
                    self._send_json(500, {"error": f"Could not start recording: {exc}"})
                else:
                    self._send_json(200, {"recording": result})
                return

            try:
                result = recorder.stop()
            except RuntimeError as exc:
                self._send_json(409, {"error": str(exc)})
            else:
                self._send_json(200, {"recording": result})

        def _read_json_object(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return None
            if length <= 0 or length > 16_384:
                self._send_json(400, {"error": "Request body must be between 1 and 16384 bytes"})
                return None
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"Invalid JSON body: {exc}"})
                return None
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "Request body must be a JSON object"})
                return None
            return payload

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Live 3D finger-motion visualizer (RGMP v2).")
    p.add_argument("--rgmp-host", default="127.0.0.1")
    p.add_argument("--rgmp-port", type=int, default=12276)
    p.add_argument("--http-host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8000)
    p.add_argument("--max-fps", type=float, default=60.0, help="Max updates/s pushed to the browser")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=HERE.parent / "data" / "captures",
        help="Directory for JSONL motion recordings",
    )
    p.add_argument("--no-browser", action="store_true", help="Don't open the page automatically")
    args = p.parse_args()
    if not math.isfinite(args.max_fps) or args.max_fps <= 0:
        p.error("--max-fps must be a finite number greater than zero")

    recorder = SessionRecorder(args.data_dir)
    state = GloveState(recorder)
    threading.Thread(
        target=rgmp_reader, args=(args.rgmp_host, args.rgmp_port, state), daemon=True
    ).start()

    server = ThreadingHTTPServer((args.http_host, args.http_port), make_handler(state, args.max_fps))
    server.daemon_threads = True
    url = f"http://{args.http_host}:{args.http_port}/"
    print(f"# visualizer at {url}  (Ctrl+C to stop)", file=sys.stderr, flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        recorder.close()


if __name__ == "__main__":
    main()
