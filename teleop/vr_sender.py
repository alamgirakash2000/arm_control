#!/usr/bin/env python3
"""
VR Tracking Sender (Windows / SteamVR)
========================================
Reads Meta Quest 3 controller tracking data via OpenVR (SteamVR)
and sends it to the relay server over HTTP at 20 Hz.

Run this on your Windows laptop with SteamVR + Quest connected.

Requirements:
    pip install openvr

Usage:
    python vr_sender.py --server http://UBUNTU_IP:8765
    python vr_sender.py --server http://192.168.1.100:8765 --hz 30
    python vr_sender.py --server http://myserver.example.com:8765   # remote

The script sends JSON frames like:
    {
      "timestamp": 1234567890.123,
      "left":  {"pos": [x,y,z], "rot": [[3x3]], "trigger": 0.5, "status": "OK"},
      "right": {"pos": [x,y,z], "rot": [[3x3]], "trigger": 0.3, "status": "OK"}
    }
"""

import argparse
import json
import sys
import time
import threading
import urllib.request
import urllib.error

# Auto-install openvr if missing
try:
    import openvr
except ImportError:
    import subprocess
    print("  openvr not found — installing ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openvr"])
    import openvr

DEFAULT_HZ = 20
CONTROLLER_RESCAN_S = 3.0


# ── OpenVR helpers ────────────────────────────────────────────────────────────

def find_controllers(vr_system):
    """Return (left_id, right_id) for tracked controllers."""
    controllers = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        try:
            if vr_system.getTrackedDeviceClass(idx) == openvr.TrackedDeviceClass_Controller:
                controllers.append(idx)
        except Exception:
            continue

    left_id = right_id = None
    for idx in controllers:
        try:
            role = vr_system.getControllerRoleForTrackedDeviceIndex(idx)
        except Exception:
            role = 0
        if role == openvr.TrackedControllerRole_LeftHand:
            left_id = idx
        elif role == openvr.TrackedControllerRole_RightHand:
            right_id = idx

    remaining = [i for i in controllers if i not in {left_id, right_id}]
    if left_id  is None and remaining: left_id  = remaining.pop(0)
    if right_id is None and remaining: right_id = remaining.pop(0)
    return left_id, right_id


def poll_controller(vr_system, cid, poses):
    """
    Poll a single controller.

    Returns (pos_list, rot_list_3x3, trigger_float, status_str).
    pos/rot are plain Python lists (JSON-serializable).
    """
    if cid is None:
        return None, None, 0.0, "NOT FOUND"

    pose = poses[cid]
    if not pose.bDeviceIsConnected:
        return None, None, 0.0, "DISCONNECTED"
    if not pose.bPoseIsValid:
        return None, None, 0.0, "NO TRACKING"

    m = pose.mDeviceToAbsoluteTracking
    pos = [float(m[0][3]), float(m[1][3]), float(m[2][3])]
    rot = [
        [float(m[0][0]), float(m[0][1]), float(m[0][2])],
        [float(m[1][0]), float(m[1][1]), float(m[1][2])],
        [float(m[2][0]), float(m[2][1]), float(m[2][2])],
    ]

    trigger = 0.0
    try:
        success, state = vr_system.getControllerState(cid)
        if success and len(state.rAxis) > 1:
            trigger = max(0.0, min(1.0, float(state.rAxis[1].x)))
    except Exception:
        pass

    return pos, rot, trigger, "OK"


# ── Non-blocking HTTP sender ─────────────────────────────────────────────────

class AsyncSender:
    """
    Background thread that sends the latest frame to the relay server.

    The main loop calls .send(payload) freely; only the freshest payload
    is sent. If the network is slow, intermediate frames are dropped
    (this is desired — the robot always wants the newest data).
    """

    def __init__(self, url, timeout=1.0):
        self._url = url
        self._timeout = timeout
        self._payload = None
        self._running = True
        self._cond = threading.Condition(threading.Lock())
        self._errors = 0
        self._sent = 0
        threading.Thread(target=self._run, daemon=True, name="sender").start()

    @property
    def errors(self):
        return self._errors

    @property
    def sent(self):
        return self._sent

    def send(self, payload_dict):
        with self._cond:
            self._payload = payload_dict
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify()

    def _run(self):
        while True:
            with self._cond:
                while self._running and self._payload is None:
                    self._cond.wait(timeout=0.1)
                if not self._running:
                    break
                payload = self._payload
                self._payload = None

            try:
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    self._url, data=data,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=self._timeout)
                self._sent += 1
                self._errors = 0
            except Exception as e:
                self._errors += 1
                if self._errors <= 3 or self._errors % 50 == 0:
                    print(f"\n  [SEND ERROR #{self._errors}] {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="VR tracking sender — reads SteamVR, sends to relay server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vr_sender.py --server http://192.168.1.100:8765
    python vr_sender.py --server http://10.0.0.5:8765 --hz 30
    python vr_sender.py --server http://myserver.com:8765
        """,
    )
    p.add_argument("--server", required=True,
                   help="Relay server URL (e.g. http://192.168.1.100:8765)")
    p.add_argument("--hz", type=int, default=DEFAULT_HZ,
                   help=f"Send rate in Hz (default: {DEFAULT_HZ})")
    p.add_argument("--prediction", type=float, default=0.010,
                   help="OpenVR pose prediction horizon in seconds (default: 0.010)")
    args = p.parse_args()

    url = args.server.rstrip("/") + "/pose"
    interval = 1.0 / args.hz

    print()
    print("=" * 55)
    print("  VR TRACKING SENDER")
    print("=" * 55)
    print(f"  Server    : {url}")
    print(f"  Send rate : {args.hz} Hz")
    print(f"  Prediction: {args.prediction} s")
    print()

    # ── Test server connectivity ──────────────────────────────────────────────
    print("  Testing server connection ...", end=" ", flush=True)
    try:
        status_url = args.server.rstrip("/") + "/status"
        resp = urllib.request.urlopen(status_url, timeout=5)
        print(f"OK ({resp.status})")
    except Exception as e:
        print(f"WARNING: {e}")
        print("  Will keep trying — make sure the server is running.")
    print()

    # ── SteamVR init ──────────────────────────────────────────────────────────
    print("  Initializing SteamVR (OpenVR) ...", end=" ", flush=True)
    try:
        vr_system = openvr.init(openvr.VRApplication_Other)
    except openvr.OpenVRError as e:
        print(f"FAILED\n  {e}")
        print("  Make sure SteamVR is running and the headset is awake.")
        sys.exit(1)
    print("OK")

    # ── Wait for controllers ──────────────────────────────────────────────────
    print("  Waiting for both controllers ...", end=" ", flush=True)
    left_id = right_id = None
    t0 = time.time()
    while left_id is None or right_id is None:
        left_id, right_id = find_controllers(vr_system)
        if time.time() - t0 > 120:
            print("\n  TIMEOUT: controllers not found in 2 minutes.")
            openvr.shutdown()
            sys.exit(1)
        time.sleep(0.5)
    print(f"OK  (L={left_id}  R={right_id})")
    print()

    # ── Start async sender ────────────────────────────────────────────────────
    sender = AsyncSender(url)
    last_rescan = time.time()
    frame = 0

    print(f"  Streaming — Ctrl+C to stop")
    print()

    try:
        while True:
            t_start = time.time()

            # Rescan controllers periodically
            if t_start - last_rescan > CONTROLLER_RESCAN_S:
                new_l, new_r = find_controllers(vr_system)
                if new_l is not None and new_r is not None:
                    if (new_l, new_r) != (left_id, right_id):
                        left_id, right_id = new_l, new_r
                        print(f"\n  [INFO] Controllers: L={left_id} R={right_id}")
                last_rescan = t_start

            # Poll tracking
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, args.prediction,
                openvr.k_unMaxTrackedDeviceCount,
            )

            payload = {"timestamp": t_start, "left": {}, "right": {}}
            for cid, hand in ((left_id, "left"), (right_id, "right")):
                pos, rot, trigger, status = poll_controller(vr_system, cid, poses)
                payload[hand] = {
                    "pos": pos if pos else [0, 0, 0],
                    "rot": rot if rot else [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "trigger": trigger,
                    "status": status,
                }

            sender.send(payload)
            frame += 1

            # Status line (every 2 seconds)
            if frame % (args.hz * 2) == 0:
                ls = payload["left"]["status"]
                rs = payload["right"]["status"]
                errs = sender.errors
                sent = sender.sent
                err_txt = f"  errs={errs}" if errs > 0 else ""
                print(f"\r  frame={frame:6d}  sent={sent:6d}  "
                      f"L={ls:<12s}  R={rs:<12s}{err_txt}    ",
                      end="", flush=True)

            # Rate limit
            elapsed = time.time() - t_start
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
    finally:
        sender.stop()
        try:
            openvr.shutdown()
            print("  SteamVR disconnected.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
