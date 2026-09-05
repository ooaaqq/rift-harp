#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RIFT_HARP_REMOTE_HOST:-connect.bjb2.seetacloud.com}"
REMOTE_PORT="${RIFT_HARP_REMOTE_PORT:-23649}"

ssh -tt \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=20 \
  -p "${REMOTE_PORT}" \
  "root@${REMOTE_HOST}" 'bash -s' <<'REMOTE'
set -u

RUN=/root/autodl-tmp/rift-harp/runs/harp-foundation-fp8-v2
SHADOW=/root/autodl-tmp/rift-harp/runs/harp-shadow-128-1630m.json

while true; do
  clear
  date

  echo "=== Progress ==="
  RUN="$RUN" python - <<'PY'
import json
import os
from pathlib import Path

def readable(value):
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))

path = Path(os.environ["RUN"]) / "events.jsonl"
if not path.exists():
    print("events.jsonl: missing")
    raise SystemExit

for line in reversed(path.read_text().splitlines()):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if event.get("type") == "train":
        print("step:", f"{event.get('global_step'):,}")
        print("requested frames:", readable(event.get("seen_requested_frames", 0)))
        print("valid frames:", readable(event.get("seen_valid_frames", 0)))
        break
else:
    print("train event: missing")
PY

  echo
  echo "=== Latest Shadow-128 Validation ==="
  if [ -f "$SHADOW" ]; then
    SHADOW="$SHADOW" python - <<'PY'
import json
import os

def readable(value):
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))

data = json.load(open(os.environ["SHADOW"]))
progress = data.get("checkpoint_progress", {})
print("checkpoint step:", f"{progress.get('global_step', 0):,}")
print("checkpoint valid frames:", readable(progress.get("seen_valid_frames", 0)))
print("checkpoint requested frames:", readable(progress.get("seen_requested_frames", 0)))
PY
  else
    echo "not available"
  fi

  sleep 30
done
REMOTE
