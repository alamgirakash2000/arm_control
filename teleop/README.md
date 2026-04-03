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


**Test server connectivity from Windows:**
```bash
curl http://10.46.34.149:8765/status
```

## Workflow

```bash
# 1. Record full pick+place demos (both cameras auto-detected)
python teleop.py --with_robot --left can0 --record ./data/demo

# 2. Annotate split points
python partition_episodes.py --src ./data/demo --annotations splits.json

# 3. Export sub-tasks
python partition_episodes.py --src ./data/demo --annotations splits.json \
    --export ./data --task-type good

# 4. Train pick policy
python policy/train.py --dataset_dir ./data/pick --output_dir ./checkpoints/pick

# 5. Train place policy
python policy/train.py --dataset_dir ./data/place_good --output_dir ./checkpoints/place_good
```