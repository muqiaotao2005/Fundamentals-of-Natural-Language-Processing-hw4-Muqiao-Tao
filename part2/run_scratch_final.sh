#!/usr/bin/env bash
set -eo pipefail
final_base="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$final_base"
final_python=/workspace/envs/a4-part2/bin/python
final_root="$final_base/scratch_runs/short_lr1e4"
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
final_bundle() {
    final_exit=$?
    trap - EXIT
    "$final_python" train_t5_scratch.py bundle --run_dir "$final_root" || true
    exit "$final_exit"
}
trap final_bundle EXIT
"$final_python" -u export_scratch_test.py --run_dir "$final_root" --device cuda \
    2>&1 | tee -a "$final_root/final_inference.log"
