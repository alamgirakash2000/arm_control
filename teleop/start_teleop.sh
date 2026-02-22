#!/bin/bash
# ==============================================================
# Piper Dual Arm Teleop Launcher
# ==============================================================
# Sets up CAN interfaces, identifies LEFT/RIGHT arms,
# then launches dual_piper.py.
#
# First run:  flashes gripper on one arm, asks you to identify.
# After that: auto-detects based on saved USB adapter serial.
#
# Usage:
#   sudo bash teleop/start_teleop.sh
#   sudo bash teleop/start_teleop.sh --reset   # Re-identify arms
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="$HOME/.piper_arms.conf"
ENV_FILE="/tmp/piper_arms.env"
BITRATE=1000000

# Reset config if requested
if [ "$1" = "--reset" ]; then
    rm -f "$CONFIG"
    echo "  Config cleared. Will re-identify arms."
    echo ""
fi

echo "========================================"
echo "  PIPER DUAL ARM CAN SETUP"
echo "========================================"
echo ""

# --- Find CAN interfaces ---
INTERFACES=$(ls /sys/class/net/ 2>/dev/null | grep "^can" | sort)
if [ -z "$INTERFACES" ]; then
    echo "  ERROR: No CAN interfaces found!"
    echo "  Make sure USB-CAN adapters are connected."
    exit 1
fi

COUNT=$(echo "$INTERFACES" | wc -w)
echo "  Found $COUNT CAN interface(s): $INTERFACES"
echo ""

# --- Get USB identifier for each interface ---
declare -A IFACE_ID

for iface in $INTERFACES; do
    # Try serial number first (most reliable across PCs)
    id=$(udevadm info -q property /sys/class/net/$iface 2>/dev/null | grep "^ID_SERIAL_SHORT=" | cut -d= -f2)
    if [ -z "$id" ]; then
        id=$(udevadm info -q property /sys/class/net/$iface 2>/dev/null | grep "^ID_SERIAL=" | cut -d= -f2)
    fi
    if [ -z "$id" ]; then
        # No serial — fall back to USB path (works on same PC only)
        id=$(udevadm info -q property /sys/class/net/$iface 2>/dev/null | grep "^ID_PATH=" | cut -d= -f2)
    fi
    if [ -z "$id" ]; then
        id="no_id_${iface}"
    fi
    IFACE_ID[$iface]=$id
    echo "  $iface → $id"
done
echo ""

# --- Set up all CAN interfaces ---
for iface in $INTERFACES; do
    ip link set $iface down 2>/dev/null || true
    ip link set $iface type can bitrate $BITRATE
    ip link set $iface up
    echo "  $iface: UP (bitrate $BITRATE)"
done
echo ""

# --- Try to match from config ---
LEFT_CAN=""
RIGHT_CAN=""

if [ -f "$CONFIG" ]; then
    SAVED_LEFT=$(grep "^LEFT_ID=" "$CONFIG" | cut -d= -f2)
    SAVED_RIGHT=$(grep "^RIGHT_ID=" "$CONFIG" | cut -d= -f2)

    for iface in $INTERFACES; do
        if [ "${IFACE_ID[$iface]}" = "$SAVED_LEFT" ]; then
            LEFT_CAN=$iface
        fi
        if [ "${IFACE_ID[$iface]}" = "$SAVED_RIGHT" ]; then
            RIGHT_CAN=$iface
        fi
    done

    if [ -n "$LEFT_CAN" ] && [ -n "$RIGHT_CAN" ]; then
        echo "  Auto-detected from saved config:"
        echo "    LEFT  = $LEFT_CAN (${IFACE_ID[$LEFT_CAN]})"
        echo "    RIGHT = $RIGHT_CAN (${IFACE_ID[$RIGHT_CAN]})"
    else
        echo "  Saved config doesn't match current adapters. Re-identifying..."
        rm -f "$CONFIG"
        LEFT_CAN=""
        RIGHT_CAN=""
    fi
fi

# --- First-time identification ---
if [ -z "$LEFT_CAN" ] || [ -z "$RIGHT_CAN" ]; then
    if [ "$COUNT" -lt 2 ]; then
        echo "  Only $COUNT CAN interface found. Need 2 for dual arm."
        echo "  Using $INTERFACES as LEFT arm only."
        LEFT_CAN=$(echo $INTERFACES | head -1)
        echo "LEFT_ID=${IFACE_ID[$LEFT_CAN]}" > "$CONFIG"
        echo "RIGHT_ID=" >> "$CONFIG"
        echo "LEFT=$LEFT_CAN" > "$ENV_FILE"
        echo "RIGHT=" >> "$ENV_FILE"
        echo ""
        echo "  LEFT arm: $LEFT_CAN"
        echo "  RIGHT arm: not connected"
        exit 0
    fi

    IFACE1=$(echo $INTERFACES | awk '{print $1}')
    IFACE2=$(echo $INTERFACES | awk '{print $2}')

    echo ""
    echo "  FIRST-TIME SETUP: Identifying arms..."
    echo "  I'll flash the gripper on one arm. Watch which one moves."
    echo ""
    echo "  Flashing gripper on $IFACE1 in 3 seconds..."
    sleep 3

    python3 -c "
from piper_sdk import C_PiperInterface_V2
import time, sys
try:
    p = C_PiperInterface_V2('$IFACE1')
    p.ConnectPort()
    time.sleep(0.3)
    for _ in range(50):
        if p.EnablePiper():
            break
        time.sleep(0.01)
    time.sleep(0.3)
    for i in range(4):
        p.GripperCtrl(50000, 1000, 0x01, 0)
        time.sleep(0.3)
        p.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.3)
    p.DisablePiper()
except Exception as e:
    print(f'  Gripper flash failed: {e}', file=sys.stderr)
" 2>&1

    echo ""
    echo "  Which arm's gripper just moved?"
    echo "    l = that was the LEFT arm"
    echo "    r = that was the RIGHT arm"
    read -p "  > " SIDE

    if [ "$SIDE" = "l" ] || [ "$SIDE" = "L" ]; then
        LEFT_CAN=$IFACE1
        RIGHT_CAN=$IFACE2
    else
        LEFT_CAN=$IFACE2
        RIGHT_CAN=$IFACE1
    fi

    # Save config
    echo "LEFT_ID=${IFACE_ID[$LEFT_CAN]}" > "$CONFIG"
    echo "RIGHT_ID=${IFACE_ID[$RIGHT_CAN]}" >> "$CONFIG"
    echo ""
    echo "  Config saved to $CONFIG"
fi

# --- Write env file for Python ---
echo "LEFT=$LEFT_CAN" > "$ENV_FILE"
echo "RIGHT=$RIGHT_CAN" >> "$ENV_FILE"

echo ""
echo "========================================"
echo "  LEFT arm:  $LEFT_CAN"
echo "  RIGHT arm: $RIGHT_CAN"
echo "========================================"
echo ""
echo "  Launching dual_piper.py ..."
echo ""
exec python3 "$SCRIPT_DIR/dual_piper.py" --left "$LEFT_CAN" --right "$RIGHT_CAN"
