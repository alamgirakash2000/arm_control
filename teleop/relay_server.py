#!/usr/bin/env python3
"""
Teleop Relay Server
====================
Lightweight HTTP relay server for VR teleoperation data.
Runs on Ubuntu (or any machine). Zero external dependencies — stdlib only.

The VR sender (Windows) POSTs controller tracking data at 20 Hz.
The robot controller (Ubuntu) GETs the latest data.

Can be run standalone, or imported and started as a background thread
inside remote_teleop.py.

Endpoints:
    POST /pose   — sender pushes a JSON frame
    GET  /pose   — receiver reads the latest frame
    GET  /status — health check (age, frame count, etc.)

Usage:
    python3 relay_server.py                  # default :8765 on all interfaces
    python3 relay_server.py --port 9000
    python3 relay_server.py --host 127.0.0.1 # localhost only
"""

import argparse
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


# ── Thread-safe pose storage ─────────────────────────────────────────────────

class PoseStore:
    """Thread-safe storage for the latest VR tracking frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self._updated_at = 0.0
        self._count = 0

    def update(self, data: dict):
        with self._lock:
            self._data = data
            self._updated_at = time.time()
            self._count += 1

    def get(self) -> tuple:
        """Returns (data_dict_or_None, updated_at_epoch, frame_count)."""
        with self._lock:
            return self._data, self._updated_at, self._count

    def age_ms(self) -> float:
        """Milliseconds since last update (-1 if no data yet)."""
        with self._lock:
            if self._updated_at == 0:
                return -1.0
            return (time.time() - self._updated_at) * 1000.0


# ── Global store (shared with importers) ─────────────────────────────────────

pose_store = PoseStore()


# ── HTTP handler ──────────────────────────────────────────────────────────────

class RelayHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/pose":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                pose_store.update(data)
                self._json_response(200, {"ok": True})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json_response(400, {"ok": False, "error": "bad JSON"})
        else:
            self._json_response(404, {"ok": False, "error": "not found"})

    def do_GET(self):
        if self.path == "/pose":
            data, updated_at, count = pose_store.get()
            if data is None:
                self.send_response(204)
                self.end_headers()
                return
            data["_server_time"] = time.time()
            data["_age_ms"] = (time.time() - updated_at) * 1000.0
            data["_count"] = count
            self._json_response(200, data)

        elif self.path == "/status":
            data, updated_at, count = pose_store.get()
            self._json_response(200, {
                "ok": True,
                "has_data": data is not None,
                "last_update": updated_at,
                "age_ms": (time.time() - updated_at) * 1000.0 if updated_at > 0 else -1,
                "frame_count": count,
            })
        else:
            self._json_response(404, {"ok": False, "error": "not found"})

    # ── helpers ───────────────────────────────────────────────────────────────

    def _json_response(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silence per-request logs for performance


# ── Server launcher ───────────────────────────────────────────────────────────

def start_server(host="0.0.0.0", port=8765, blocking=True):
    """
    Start the relay HTTP server.

    Args:
        host:     bind address ("0.0.0.0" = all interfaces)
        port:     TCP port
        blocking: if False, runs in a daemon thread and returns immediately
    """
    server = HTTPServer((host, port), RelayHandler)
    if blocking:
        print(f"  Relay server listening on {host}:{port}")
        print(f"  POST /pose   — sender pushes tracking data")
        print(f"  GET  /pose   — receiver reads latest frame")
        print(f"  GET  /status — health check")
        print()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True,
                             name="relay-server")
        t.start()
        return server


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Teleop relay server")
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address (default: 0.0.0.0 = all interfaces)")
    p.add_argument("--port", type=int, default=8765,
                   help="TCP port (default: 8765)")
    args = p.parse_args()

    print()
    print("=" * 55)
    print("  TELEOP RELAY SERVER")
    print("=" * 55)
    print()
    start_server(host=args.host, port=args.port, blocking=True)


if __name__ == "__main__":
    main()
