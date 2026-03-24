#!/bin/bash
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    echo "Run this script with sudo." >&2
    exit 1
fi

cat >/etc/modprobe.d/nvidia-stability.conf <<'EOF'
# Work around long-running CUDA instability on Linux desktop systems:
# - Disable HMM in nvidia_uvm
# - Disable GSP firmware in the nvidia module
options nvidia_uvm uvm_disable_hmm=1
options nvidia NVreg_EnableGpuFirmware=0
EOF

update-initramfs -u

if command -v update-grub >/dev/null 2>&1; then
    update-grub
fi

echo
echo "Applied NVIDIA stability workarounds."
echo "Reboot is required before the new module parameters take effect."
