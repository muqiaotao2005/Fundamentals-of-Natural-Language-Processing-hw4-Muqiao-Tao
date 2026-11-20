#!/usr/bin/env bash
# Run on the existing RunPod, with the repository stored under /workspace.
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /workspace/miniforge3/etc/profile.d/conda.sh
if [ ! -x /workspace/envs/a4-part2/bin/python ]; then
    conda create -p /workspace/envs/a4-part2 python=3.11 pip -y
fi
conda activate /workspace/envs/a4-part2
conda env config vars set HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'No CUDA GPU detected; training was not started.'
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
print('GPU calculation:', (torch.ones(3, device='cuda') * 2).tolist())
PY

python -u train_t5.py --finetune --experiment_name experiment \
    --learning_rate 3e-4 --batch_size 16 --test_batch_size 16 \
    --max_n_epochs 20 --patience_epochs 4 \
    --scheduler_type linear --num_warmup_epochs 1 \
    --num_beams 1 --max_new_tokens 512 --seed 42 \
    2>&1 | tee t5_baseline.log
