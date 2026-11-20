#!/usr/bin/env bash
# Launch once in the background; the existing /workspace environment is reused.
set -eo pipefail
autopilot_base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$autopilot_base"
autopilot_python=/workspace/envs/a4-part2/bin/python
autopilot_root="$autopilot_base/scratch_runs/short_lr1e4"
if [ ! -x "$autopilot_python" ] || [ ! -f "$autopilot_root/pilot/latest.pt" ]; then
    echo 'Expected the existing a4-part2 environment and short_lr1e4 pilot checkpoint.'
    exit 1
fi
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
nohup "$autopilot_python" -u "$autopilot_base/scratch_autopilot.py" \
    --run_dir "$autopilot_root" --max_epochs 60 --min_epochs 30 \
    --patience 3 --min_delta 0.002 \
    >> "$autopilot_root/autopilot.log" 2>&1 < /dev/null &
autopilot_pid=$!
echo "Autopilot launched (PID $autopilot_pid). It waits if the current pilot is still running."
echo "Log: $autopilot_root/autopilot.log"
echo "Status: $autopilot_root/autopilot_status.json"
echo 'The process runs independently of this terminal. Keep the Pod running.'
