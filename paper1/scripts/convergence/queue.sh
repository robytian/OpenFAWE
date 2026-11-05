#!/bin/bash
# Sequential job queue: each line of the job file is "NAME ARGS..." for run_history.py.
# Usage: queue.sh JOBFILE  (runs in local/accel-20261001/runs, one Numba thread per job)
set -u
JOBS="$(realpath "$1")"
TASK="$(cd "$(dirname "$0")" && pwd)"
cd "$TASK/../../../local/accel-20261001/runs"
PY="${OPENFAWE_PY:-python3}"
while read -r name rest; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  if [ -e "$name/history.json" ]; then echo "skip $name"; continue; fi
  rm -rf "$name"
  NUMBA_NUM_THREADS=1 "$PY" "$TASK/run_history.py" "$name" $rest > "$name.log" 2>&1
  tail -n 1 "$name.log"
done < "$JOBS"
