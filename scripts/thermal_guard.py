"""Pause a workload before the laptop's firmware cuts power.

This machine (HP Pavilion Gaming 15-ec2xxx, Ryzen 5 5600H + RTX 3050) hard
powers off under sustained load: the CPU sits at 95-98 C against a 98 C trip
point, `thermald` reports the platform unsupported, and ACPI advertises an
invalid critical threshold, so the kernel cannot shut down gracefully. The
embedded controller cuts power with nothing written to the logs.

The fix that needs no root and no vendor tooling: watch the die temperature and
SIGSTOP the workload when it approaches the limit, SIGCONT it once the chassis
has cooled. A paused process keeps its memory, its CUDA context and its open
files, so a long training run simply takes longer instead of dying.

Hysteresis matters. Pausing and resuming at the same temperature would thrash
the process dozens of times a minute; a gap between the pause and resume points
lets the heatsink actually shed heat.

    python scripts/thermal_guard.py --match run_experiment
    python scripts/thermal_guard.py --match "finish_experiment|run_experiment" \
        --pause-at 90 --resume-at 80
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import time
from pathlib import Path

#: k10temp's Tctl. Read from sysfs rather than shelling out to `sensors`, which
#: costs ~50 ms per sample and would itself add load.
DEFAULT_SENSOR = "/sys/class/hwmon/hwmon4/temp1_input"


def find_sensor() -> Path:
    """Locate Tctl, falling back across hwmon numbering that can change on boot."""
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = (hwmon / "name")
        if name.exists() and name.read_text().strip() == "k10temp":
            probe = hwmon / "temp1_input"
            if probe.exists():
                return probe
    fallback = Path(DEFAULT_SENSOR)
    if fallback.exists():
        return fallback
    raise SystemExit("could not find the k10temp sensor")


def cpu_temp(sensor: Path) -> float:
    try:
        return int(sensor.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return 0.0


def gpu_temp() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return float(out[0]) if out else 0.0
    except Exception:
        return 0.0


def matching_pids(pattern: str, exclude: set[int], skip: str | None = None) -> list[int]:
    """PIDs whose command line matches, excluding this guard and its shell.

    Self-matching is the classic failure here: a pattern that appears in the
    guard's own command line would make it pause itself.
    """
    pids = []
    regex = re.compile(pattern)
    # Some matching processes must never be stopped. A SIGSTOP costs a compute
    # job nothing but wall-clock, while an upload holds an open socket the far
    # end will close: pausing `kaggle competitions submit` mid-transfer can fail
    # the submission outright. Those jobs also use almost no CPU, so they are
    # not what is heating the machine.
    skip_regex = re.compile(skip) if skip else None
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid in exclude:
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "thermal_guard" in cmdline:
            continue
        if skip_regex is not None and skip_regex.search(cmdline):
            continue
        if regex.search(cmdline):
            pids.append(pid)
    return pids


def is_stopped(pid: int) -> bool:
    """True if the process is sitting in SIGSTOP.

    A guard that is killed while the workload is paused leaves it paused
    forever, and the supervisor's replacement has no memory of what it stopped.
    Reading the state back from /proc is what lets the replacement clean up.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # The command field can contain spaces and parentheses; state follows it.
    return stat.rpartition(")")[2].split()[0] in {"T", "t"}


def signal_all(pids: list[int], sig: signal.Signals) -> int:
    sent = 0
    for pid in pids:
        try:
            import os

            os.kill(pid, sig)
            sent += 1
        except (ProcessLookupError, PermissionError):
            continue
    return sent


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True,
                    help="regex matched against process command lines")
    ap.add_argument("--exclude", default=None,
                    help="regex for command lines to leave alone even when they match")
    ap.add_argument("--pause-at", type=float, default=90.0,
                    help="pause the workload at or above this CPU temp (C)")
    ap.add_argument("--resume-at", type=float, default=80.0,
                    help="resume below this temp; the gap prevents thrashing")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--wait", action="store_true",
                    help="keep running when nothing matches, for unattended pipelines")
    args = ap.parse_args()

    if args.resume_at >= args.pause_at:
        raise SystemExit("--resume-at must be below --pause-at, or it will thrash")

    sensor = find_sensor()
    exclude = {os.getpid(), os.getppid()}
    paused = False
    waiting = False
    pauses = 0
    stalled = 0.0
    start = time.time()

    def emit(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if args.log:
            with args.log.open("a") as fh:
                fh.write(line + "\n")

    emit(f"guarding /{args.match}/ — pause at {args.pause_at:.0f}C, "
         f"resume at {args.resume_at:.0f}C, sensor {sensor}")

    def shutdown(_sig, _frame):
        held = [p for p in matching_pids(args.match, exclude, args.exclude) if is_stopped(p)]
        if held:
            signal_all(held, signal.SIGCONT)
            emit(f"resumed {len(held)} process(es) before exiting")
        emit(f"stopped after {pauses} pauses, {stalled:.0f}s stalled")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pause_began = 0.0
    while True:
        temp = cpu_temp(sensor)
        pids = matching_pids(args.match, exclude, args.exclude)

        if not pids:
            if paused:
                paused, pause_began = False, 0.0  # the workload exited while paused
            if not args.wait:
                emit(f"no matching process (cpu {temp:.0f}C) — exiting")
                return 0
            if not waiting:
                emit(f"no matching process (cpu {temp:.0f}C) — waiting")
                waiting = True
            time.sleep(args.interval)
            continue
        waiting = False

        if temp >= args.pause_at:
            # Pause on every hot poll, not only on the transition into the hot
            # state. A job that starts while the guard is already holding the
            # workload would otherwise run completely unguarded, and because it
            # keeps the die hot the guard never cools back down to notice it --
            # which is exactly how this machine spent 88 minutes at 92C with
            # every process it knew about frozen.
            fresh = [pid for pid in pids if not is_stopped(pid)]
            if fresh:
                n = signal_all(fresh, signal.SIGSTOP)
                if not paused:
                    paused, pause_began = True, time.time()
                pauses += 1
                emit(f"cpu {temp:.1f}C >= {args.pause_at:.0f}C — paused {n} process(es)")
        elif temp <= args.resume_at:
            # Resume by observed state rather than by what this guard stopped,
            # so a replacement guard cleans up after the one it replaced.
            held = [pid for pid in pids if is_stopped(pid)]
            if held:
                n = signal_all(held, signal.SIGCONT)
                began = pause_began or time.time()
                stalled += time.time() - began
                emit(f"cpu {temp:.1f}C <= {args.resume_at:.0f}C — resumed {n} process(es) "
                     f"(stalled {time.time() - began:.0f}s)")
            paused, pause_began = False, 0.0

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
