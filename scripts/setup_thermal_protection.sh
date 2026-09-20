#!/usr/bin/env bash
# Reduce sustained heat output on this laptop so long runs stop tripping the
# firmware's power cut.
#
# Background: HP Pavilion Gaming 15-ec2xxx, Ryzen 5 5600H (45 W) + RTX 3050
# (60 W). Under sustained load the CPU sits at 95-99 C against a 98 C trip
# point. `thermald` reports the platform unsupported and ACPI advertises an
# invalid critical threshold, so the embedded controller cuts power outright
# with nothing written to the logs.
#
# thermal_guard.py prevents the crash by pausing the workload, but it ends up
# cycling every few seconds because the CPU reaches 90 C within about two
# seconds of full load. These settings lower the heat at the source so the
# guard rarely has to intervene.
#
# Everything here is reversible and survives nothing: re-run on reboot, or use
# --persist to install a systemd unit.
#
#   sudo bash scripts/setup_thermal_protection.sh
#   sudo bash scripts/setup_thermal_protection.sh --revert
#   sudo bash scripts/setup_thermal_protection.sh --persist

set -euo pipefail

GPU_WATTS=${GPU_WATTS:-40}      # default limit is 60 W
GOVERNOR=powersave              # `performance` pins boost clocks and runs hot
ACTION=${1:-apply}

if [[ $EUID -ne 0 ]]; then
  echo "needs root: sudo bash $0 ${ACTION}" >&2
  exit 1
fi

show_state() {
  echo "  governor : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "  gpu limit: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null || echo n/a)"
  local t
  t=$(cat /sys/class/hwmon/hwmon*/temp1_input 2>/dev/null | sort -rn | head -1)
  echo "  cpu temp : $(( ${t:-0} / 1000 )) C"
}

case "$ACTION" in
  --revert|revert)
    echo "reverting to stock settings"
    for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      echo performance > "$c" 2>/dev/null || true
    done
    nvidia-smi -pm 1 >/dev/null 2>&1 || true
    nvidia-smi -pl 60 >/dev/null 2>&1 || true
    systemctl disable --now thermal-guard.service 2>/dev/null || true
    show_state
    exit 0
    ;;

  --persist|persist)
    # Re-apply on every boot. The settings themselves are not sticky.
    cat > /etc/systemd/system/thermal-protection.service <<EOF
[Unit]
Description=Lower sustained CPU/GPU power on this chassis
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/env bash $(readlink -f "$0") apply
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now thermal-protection.service
    echo "installed thermal-protection.service (re-applies on boot)"
    show_state
    exit 0
    ;;
esac

echo "before:"
show_state

# 1. CPU governor. On amd_pstate `powersave` still boosts on demand; it simply
#    stops holding maximum clocks during sustained load, which is where the
#    heat comes from.
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo "$GOVERNOR" > "$c" 2>/dev/null || true
done

# 2. GPU power cap. Persistence mode first, or the limit is dropped whenever no
#    process holds the device.
nvidia-smi -pm 1 >/dev/null 2>&1 || true
if nvidia-smi -pl "$GPU_WATTS" >/dev/null 2>&1; then
  echo "  gpu capped at ${GPU_WATTS} W"
else
  echo "  WARNING: GPU power cap rejected (locked on some laptop vBIOSes)"
fi

echo "after:"
show_state
cat <<'NOTE'

Still recommended, in rough order of effect:
  * raise the laptop so the underside intake is not against the desk
  * clean the fans and heatsink - these chassis clog and lose most of their
    headroom to dust
  * run long jobs with scripts/thermal_guard.py as a backstop
  * check for a BIOS update: the invalid ACPI critical threshold
    (-274000) is a firmware bug, and a fixed table would let the kernel
    throttle and shut down gracefully instead of losing power outright
NOTE
