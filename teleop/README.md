# Quest Teleop — Dual Piper Arms

Teleoperate two hanging Piper arms using Meta Quest 3 controllers.

```
Quest 3 → SteamVR (Windows laptop) → HTTP → Ubuntu (robot control)
```

---

## Remote Teleop Setup (Step by Step)

### One-time setup

**Windows laptop:**
```bash
pip install openvr
```
Also install SteamVR + Meta Quest app from Steam.

**Ubuntu PC:**
```bash
pip install opencv-python   # optional, for status overlay window
```

---

### How to run

**Step 1 — Ubuntu: start CAN** (once per session)
```bash
sudo bash ../teleop/start_teleop.sh
```

**Step 2 — Ubuntu: start robot controller**
```bash
python3 remote_teleop.py                                    # simulation (safe, no robot)
python3 remote_teleop.py --with_robot                       # real robot
python3 remote_teleop.py --with_robot --left can0 --right can1   # explicit CAN ports
```
This starts an HTTP server on port `8765` and waits for VR data.

**Step 3 — Windows laptop: connect Quest**
1. Open SteamVR
2. Connect Quest 3 via Air Link or Quest Link
3. Confirm both controllers show as tracked in SteamVR

**Step 4 — Windows laptop: start sender**
```bash
python vr_sender.py --server http://10.46.34.149:8765
```

**Step 5 — Calibrate**

The Ubuntu terminal will show a countdown. Hold both controllers at the robot home position and keep still.

**Step 6 — Teleoperate**

Move controllers → robot follows. Trigger = gripper. Ctrl+C to stop.

---

## Options

| Flag | Default | What it does |
|---|---|---|
| `--with_robot` | off | Actually move the robot (without this, simulation only) |
| `--left can0` | auto | CAN interface for left arm |
| `--right can1` | auto | CAN interface for right arm |
| `--scale 0.5` | `1.0` | Motion scale (smaller = less range, safer) |
| `--speed 15` | `20` | Robot speed % (start low) |
| `--smoothing 0.6` | `0.4` | Smoothing (higher = smoother but laggier) |
| `--invert-gripper` | off | Flip trigger direction |
| `--port 9000` | `8765` | HTTP server port |

---

## Files

| File | Where | What |
|---|---|---|
| `remote_teleop.py` | Ubuntu | Robot controller + built-in HTTP server |
| `vr_sender.py` | Windows | Reads SteamVR, sends tracking data to Ubuntu |
| `relay_server.py` | Ubuntu | Standalone server (optional, for testing) |
| `quest_receiver.py` | Same PC | Local mode (SteamVR + robot on same machine) |
| `piper_core.py` | Ubuntu | Arm math (FK, IK) and robot interface |
| `dual_piper.py` | Ubuntu | Interactive keyboard control |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Sender shows "Send error" | Check Ubuntu IP; check firewall allows port 8765 |
| Ubuntu stuck on "Waiting for VR sender" | Start `vr_sender.py` on Windows; check network |
| Robot freezes mid-operation | Data went stale; check vr_sender + network |
| Arms move wrong direction | Edit `AXIS_MAP_L` / `AXIS_MAP_R` in the script |
| `IK failed` | Reduce `--scale` |
| Jittery | Increase `--smoothing` to 0.6 |

**Test server connectivity from Windows:**
```bash
curl http://10.46.34.149:8765/status
```
