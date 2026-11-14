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
import os
import socket
import struct
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


def detect_hand(definition: dict):
    """'left' / 'right' from the definition's device_info, or None if it doesn't say."""
    info = definition.get("device_info")
    if not isinstance(info, dict):
        return None
    hand = info.get("hand")
    if isinstance(hand, str) and hand.lower() in ("left", "right"):
        return hand.lower()
    is_right = info.get("is_right_glove")
    if isinstance(is_right, bool):
        return "right" if is_right else "left"
    return None


class GloveState:
    """Latest frame per group for every connected glove, shared between the RGMP reader and HTTP clients."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.seq = 0
        self.status = "starting"
        # device_id -> {"hand": "left"|"right", "groups": {name: {"t", "s"}}, "rates": {}, "count": {}}
        self.devices: dict[int, dict] = {}
        self._rate_t0 = time.monotonic()

    def set_status(self, status: str) -> None:
        with self.cond:
            if status != self.status:
                print(f"# {status}", file=sys.stderr, flush=True)
            self.status = status
            self.seq += 1
            self.cond.notify_all()

    def _streaming_status(self) -> str:
        if not self.devices:
            return "device disconnected, waiting"
        return "streaming " + ", ".join(
            f"{d['hand']} glove {dev_id}" for dev_id, d in sorted(self.devices.items())
        )

    def add_device(self, device_id: int, hand) -> None:
        with self.cond:
            if hand is None:
                # Glove doesn't report its side: take the free slot, right first.
                taken = {d["hand"] for i, d in self.devices.items() if i != device_id}
                hand = "left" if "right" in taken and "left" not in taken else "right"
            self.devices[device_id] = {"hand": hand, "groups": {}, "rates": {}, "count": {}}
        self.set_status(self._streaming_status())

    def remove_device(self, device_id) -> None:
        with self.cond:
            if device_id is None:
                self.devices.clear()
            else:
                self.devices.pop(device_id, None)
        self.set_status(self._streaming_status())

    def clear_devices(self) -> None:
        with self.cond:
            self.devices.clear()

    def update_group(self, device_id: int, name: str, ts: int, streams: dict) -> None:
        with self.cond:
            dev = self.devices.get(device_id)
            if dev is None:
                return
            dev["groups"][name] = {"t": ts, "s": streams}
            dev["count"][name] = dev["count"].get(name, 0) + 1
            now = time.monotonic()
            if now - self._rate_t0 >= 1.0:
                dt = now - self._rate_t0
                for d in self.devices.values():
                    d["rates"] = {k: round(v / dt, 1) for k, v in d["count"].items()}
                    d["count"] = {}
                self._rate_t0 = now
            self.seq += 1
            self.cond.notify_all()

    def snapshot_json(self) -> str:
        hands = {
            d["hand"]: {"device_id": dev_id, "groups": d["groups"], "rates": d["rates"]}
            for dev_id, d in sorted(self.devices.items())
        }
        # Top-level device_id/rates/groups describe one glove (the right one if present),
        # for pages that only show a single hand.
        primary = hands.get("right") or hands.get("left") or {}
        return json.dumps(
            {
                "status": self.status,
                "device_id": primary.get("device_id"),
                "rates": primary.get("rates", {}),
                "groups": primary.get("groups", {}),
                "hands": hands,
            },
            separators=(",", ":"),
        )


def rgmp_reader(host: str, port: int, state: GloveState) -> None:
    """Connect to the RGMP server, decode frames into state; reconnect forever."""
    while True:
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                state.clear_devices()
                state.set_status(f"connected to {host}:{port}, waiting for definition")
                # Per device: group_id -> stream descriptors, and group_id -> group name.
                lookups: dict[int, dict] = {}
                names: dict[int, dict[int, str]] = {}
                while True:
                    prefix, length = struct.unpack(
                        OUTER_HEADER_FMT, recv_exact(sock, OUTER_HEADER_SIZE)
                    )
                    payload = recv_exact(sock, length)

                    if prefix == MSG_PREFIX_DEFINITION:
                        definition = json.loads(payload.decode("utf-8"))
                        dev_id = definition.get("device_id")
                        lookups[dev_id] = build_group_lookup(definition, None, None, None)
                        names[dev_id] = {
                            i: g.get("name", f"group_{i}")
                            for i, g in enumerate(definition.get("groups", []))
                        }
                        state.add_device(dev_id, detect_hand(definition))

                    elif prefix == MSG_PREFIX_DATA:
                        if len(payload) < DATA_FRAME_HEADER_SIZE:
                            continue
                        dev_id, group_id, ts = struct.unpack_from(
                            DATA_FRAME_HEADER_FMT, payload, 0
                        )
                        descs = lookups.get(dev_id, {}).get(group_id)
                        if not descs:
                            continue
                        body = memoryview(payload)[DATA_FRAME_HEADER_SIZE:]
                        streams: dict[str, dict] = {}
                        for d in descs:
                            vals = struct.unpack_from(d.struct_fmt, body, d.offset)
                            vals = [round(v, 5) if isinstance(v, float) else v for v in vals]
                            streams.setdefault(d.target_frame, {})[stream_key(d)] = vals
                        group = names[dev_id].get(group_id, f"group_{group_id}")
                        state.update_group(dev_id, group, ts, streams)

                    elif prefix == MSG_PREFIX_DISCONNECT:
                        dev_id = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else None
                        lookups.pop(dev_id, None)
                        names.pop(dev_id, None)
                        state.remove_device(dev_id)
        except (OSError, ConnectionError) as exc:
            state.clear_devices()
            state.set_status(f"waiting for RGMP stream on {host}:{port} ({exc})")
            time.sleep(1.0)


PAGES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/music": "music.html",
    "/music.html": "music.html",
}


def make_handler(state: GloveState, max_fps: float):
    min_interval = 1.0 / max_fps

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the console quiet
            pass

        def do_GET(self) -> None:
            page = PAGES.get(self.path.split("?")[0])
            if page:
                body = (HERE / page).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
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

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Live 3D finger-motion visualizer (RGMP v2).")
    p.add_argument("--rgmp-host", default="127.0.0.1")
    p.add_argument("--rgmp-port", type=int, default=12276)
    p.add_argument("--http-host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8000)
    p.add_argument("--max-fps", type=float, default=60.0, help="Max updates/s pushed to the browser")
    p.add_argument("--no-browser", action="store_true", help="Don't open the page automatically")
    args = p.parse_args()

    state = GloveState()
    threading.Thread(
        target=rgmp_reader, args=(args.rgmp_host, args.rgmp_port, state), daemon=True
    ).start()

    server = ThreadingHTTPServer((args.http_host, args.http_port), make_handler(state, args.max_fps))
    server.daemon_threads = True
    url = f"http://{args.http_host}:{args.http_port}/"
    print(f"# visualizer at {url}  |  chords at {url}music  (Ctrl+C to stop)", file=sys.stderr, flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
