#!/usr/bin/env bash
# Keep thermal_guard.py alive.
#
# The guard died once mid-run -- probably taken out with the desktop session --
# and the machine then ran at 96 C, unprotected, for eighteen minutes against a
# 98 C trip point. A protection that silently stops protecting is worse than
# none, because the run looks healthy right up until the power cuts.
#
# This restarts it whenever it is missing, and logs every restart so the gaps
# are visible afterwards rather than invisible.
#
#   setsid bash scripts/guard_supervisor.sh > ~/sonics/supervisor.log 2>&1 &

PY=/home/laksh/Downloads/rmml/.venv/bin/python
GUARD=/home/laksh/Downloads/rmml/code/scripts/thermal_guard.py
LOG=/home/laksh/sonics/thermal_guard.log
# Match any script in this repo rather than an enumerated list. The list went
# stale the moment a new script was added -- generator_split_eval.py ran
# unguarded at 97 C against a 98 C trip because it was not named here.
#
# Matched on the *relative* path, because that is how the scripts are invoked:
# an absolute-path pattern matched nothing and left the workload unguarded.
MATCH='scripts/[a-z_]+\.py'

restarts=0
while true; do
  # Count real guard processes, not this supervisor and not a matching shell:
  # `pgrep -f thermal_guard` would match both and report a dead guard as alive.
  alive=$(ps -eo pid,args | grep -c "[t]hermal_guard\.py --match")

  if [ "$alive" -eq 0 ]; then
    restarts=$((restarts + 1))
    echo "$(date +%H:%M:%S) guard missing — restart #$restarts"
    setsid "$PY" -u "$GUARD" --match "$MATCH" \
      --pause-at 88 --resume-at 78 --interval 1.5 --wait --log "$LOG" \
      >> /home/laksh/sonics/guard_restarts.out 2>&1 < /dev/null &
    disown 2>/dev/null
    sleep 10
  fi
  sleep 15
done
