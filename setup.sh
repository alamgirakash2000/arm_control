#!/bin/bash
# ==============================================================
# Piper Arm CAN Setup
# ==============================================================
# Resets all CAN interfaces, brings them up fresh, and asks
# you to verify which arm is LEFT/RIGHT.
#
# Usage:
#   sudo bash setup.sh
# ==============================================================

CONFIG="$HOME/.piper_arms.conf"
ENV_FILE="/tmp/piper_arms.env"
BITRATE=1000000

echo "========================================"
echo "  PIPER ARM CAN SETUP"
echo "========================================"
echo ""

# --- Clear old config ---
rm -f "$CONFIG" "$ENV_FILE"

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

# --- Reset and bring up all CAN interfaces ---
echo "  Resetting all CAN interfaces..."
for iface in $INTERFACES; do
    ip link set $iface down 2>/dev/null || true
done
sleep 1

for iface in $INTERFACES; do
    ip link set $iface type can bitrate $BITRATE
    ip link set $iface up
    echo "  $iface: UP (bitrate $BITRATE)"
done
echo ""

# --- Get USB identifier for each interface ---
declare -A IFACE_ID
for iface in $INTERFACES; do
    id=$(udevadm info -q property /sys/class/net/$iface 2>/dev/null | grep "^ID_SERIAL_SHORT=" | cut -d= -f2)
    if [ -z "$id" ]; then
        id=$(udevadm info -q property /sys/class/net/$iface 2>/dev/null | grep "^ID_SERIAL=" | cut -d= -f2)
    fi
    if [ -z "$id" ]; then
        id="no_id_${iface}"
    fi
    IFACE_ID[$iface]=$id
    echo "  $iface → $id"
done
echo ""

# --- Single arm ---
if [ "$COUNT" -lt 2 ]; then
    SINGLE=$(echo $INTERFACES | awk '{print $1}')
    echo "  Only 1 CAN interface. Which arm is this?"
    echo "    l = LEFT arm"
    echo "    r = RIGHT arm"
    read -p "  > " SIDE

    if [ "$SIDE" = "l" ] || [ "$SIDE" = "L" ]; then
        LEFT_CAN=$SINGLE
        RIGHT_CAN=""
    else
        LEFT_CAN=""
        RIGHT_CAN=$SINGLE
    fi

    LEFT_ID=""
    RIGHT_ID=""
    [ -n "$LEFT_CAN" ] && LEFT_ID="${IFACE_ID[$LEFT_CAN]}"
    [ -n "$RIGHT_CAN" ] && RIGHT_ID="${IFACE_ID[$RIGHT_CAN]}"
    echo "LEFT_ID=$LEFT_ID" > "$CONFIG"
    echo "RIGHT_ID=$RIGHT_ID" >> "$CONFIG"
    echo "LEFT=${LEFT_CAN:-}" > "$ENV_FILE"
    echo "RIGHT=${RIGHT_CAN:-}" >> "$ENV_FILE"

    echo ""
    echo "========================================"
    [ -n "$LEFT_CAN" ]  && echo "  LEFT arm:  $LEFT_CAN"
    [ -n "$RIGHT_CAN" ] && echo "  RIGHT arm: $RIGHT_CAN"
    echo "========================================"
    exit 0
fi

# --- Dual arm: flash gripper to identify ---
IFACE1=$(echo $INTERFACES | awk '{print $1}')
IFACE2=$(echo $INTERFACES | awk '{print $2}')

echo "  Identifying arms — watch which gripper moves..."
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
echo "LEFT=$LEFT_CAN" > "$ENV_FILE"
echo "RIGHT=$RIGHT_CAN" >> "$ENV_FILE"

echo ""
echo "========================================"
echo "  LEFT arm:  $LEFT_CAN"
echo "  RIGHT arm: $RIGHT_CAN"
echo "========================================"
echo ""
echo "  Config saved. Ready to go."
