#!/usr/bin/env bash
# Existing RunPod environment; all outputs stay on /workspace.
set -eo pipefail
scratch_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$scratch_script_dir"
scratch_stage="${1:-all}"
if [ "$#" -gt 0 ]; then shift; fi
case "$scratch_stage" in
    all|preflight|overfit|pilot) ;;
    *) echo 'Usage: bash run_scratch_short.sh {all|preflight|overfit|pilot} [options]'; exit 1 ;;
esac
scratch_run_dir="${SCRATCH_RUN_DIR:-$scratch_script_dir/scratch_runs/short_v1}"
scratch_options=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run_dir)
            if [ "$#" -lt 2 ]; then echo '--run_dir requires a path'; exit 1; fi
            scratch_run_dir="$2"; shift 2 ;;
        --run_dir=*) scratch_run_dir="${1#--run_dir=}"; shift ;;
        *) scratch_options+=("$1"); shift ;;
    esac
done

source /workspace/miniforge3/etc/profile.d/conda.sh
if [ ! -x /workspace/envs/a4-part2/bin/python ]; then
    conda create -p /workspace/envs/a4-part2 python=3.11 pip -y
fi
conda activate /workspace/envs/a4-part2
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
if ! python -c 'import torch, transformers, tokenizers, numpy, sentencepiece, tqdm; assert torch.__version__.split("+")[0] == "2.1.2" and transformers.__version__ == "4.40.0" and tokenizers.__version__ == "0.19.1" and numpy.__version__ == "1.26.0"'; then
    python -m pip install -r requirements.txt
fi
python -m pip check
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable; GPU short run was not started.'
print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)
print('CUDA check:', (torch.ones(3, device='cuda') * 2).tolist())
PY

mkdir -p "$scratch_run_dir/logs"
scratch_bundle_on_exit() {
    scratch_exit_status=$?
    trap - EXIT
    python train_t5_scratch.py bundle --run_dir "$scratch_run_dir" || true
    exit "$scratch_exit_status"
}
trap scratch_bundle_on_exit EXIT
scratch_run_stage() {
    local scratch_name="$1"
    python -u train_t5_scratch.py "$scratch_name" --device cuda \
        --run_dir "$scratch_run_dir" "${scratch_options[@]}" \
        2>&1 | tee "$scratch_run_dir/logs/${scratch_name}_$(date +%Y%m%d_%H%M%S).log"
}
if [ "$scratch_stage" = all ]; then
    scratch_run_stage preflight
    scratch_run_stage overfit
    scratch_run_stage pilot
else
    scratch_run_stage "$scratch_stage"
fi
