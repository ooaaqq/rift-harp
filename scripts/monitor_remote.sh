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

  echo
  echo "=== GPU ==="
  nvidia-smi \
    --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.reserved,memory.total,power.draw \
    --format=csv,noheader || true

  echo
  echo "=== Processes ==="
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader || true

  echo
  echo "=== Training ==="
  RUN="$RUN" python - <<'PY'
import json
import os
from pathlib import Path

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
        for key in (
            "global_step",
            "seen_requested_frames",
            "seen_valid_frames",
            "seen_voiced_frames",
            "loss",
            "grad_norm",
            "frames_per_second",
            "clipped_updates",
            "lambda_floor_fraction",
            "q_floor_fraction",
        ):
            print(f"{key}: {event.get(key)}")
        break
else:
    print("train event: missing")
PY

  echo
  echo "=== Latest Checkpoints ==="
  find "$RUN" -maxdepth 1 -name '*.pt' -printf '%T@ %f\n' 2>/dev/null \
    | sort -rn | head -5 || true

  echo
  echo "=== Shadow-128 ==="
  if [ -f "$SHADOW" ]; then
    SHADOW="$SHADOW" python - <<'PY'
import json
import os

data = json.load(open(os.environ["SHADOW"]))
print("artifact:", data.get("artifact_type"))
for model in ("harp_ema_null", "harp_ema_correct"):
    print(model)
    for length in ("256", "512", "768"):
        row = data["models"][model][length]
        print(
            length,
            "mean=", round(row["mean_active_raw_mse"], 6),
            "median=", round(row["median_active_raw_mse"], 6),
            "catastrophe=", row["catastrophe_count"],
        )
PY
  else
    echo "not available"
  fi

  sleep 30
done
REMOTE
