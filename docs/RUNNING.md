# Running the code

## Course data

Run from the repository root:

```bash
python scripts/fetch_course_data.py
```

The downloader checks the pinned file hashes and preserves existing files.

## Part 1 environment

Use a separate Python 3.11 environment for each part. Run from the repository root:

```bash
conda create -n dsga1011-hw4-part1 python=3.11 -y
conda activate dsga1011-hw4-part1
python -m pip install -r part1/requirements.txt
python -m nltk.downloader wordnet punkt averaged_perceptron_tagger
cd part1
```

### Baseline training and evaluation

```bash
python -u main.py --train --eval --learning_rate 5e-5 --num_epochs 3 --batch_size 8
python -u main.py --eval_transformed --transform typos --transform_prob 0.35 --transform_seed 0
```

The checkpoint is saved in `out/`; outputs are `out_original.txt` and
`out_transformed.txt`.

### Augmented training and evaluation

```bash
python -u main.py --train_augmented --eval --eval_transformed \
  --transform typos --transform_prob 0.35 --transform_seed 0
```

The checkpoint is saved in `out_augmented/`; outputs are
`out_augmented_original.txt` and `out_augmented_transformed.txt`.
To evaluate a saved augmented checkpoint:

```bash
python -u main.py --eval --eval_transformed --model_dir out_augmented \
  --transform typos --transform_prob 0.35 --transform_seed 0
```

## Part 2 environment

Run from the repository root:

```bash
conda create -n dsga1011-hw4-part2 python=3.11 -y
conda activate dsga1011-hw4-part2
python -m pip install -r part2/requirements.txt
python -m pip check
cd part2
```

Training commands require a CUDA GPU. The direct Python commands use the active
environment. RunPod shell launchers expect `/workspace/miniforge3` and
`/workspace/envs/a4-part2`.

### Fine-tuning

```bash
python -u train_t5.py --finetune --experiment_name experiment \
  --learning_rate 3e-4 --batch_size 16 --test_batch_size 16 \
  --max_n_epochs 20 --patience_epochs 4 \
  --scheduler_type linear --num_warmup_epochs 1 \
  --num_beams 1 --max_new_tokens 512 --seed 42
```

Saved experiment: `checkpoints/ft_experiments/experiment/`.
For inference from the selected checkpoint:

```bash
python -u train_t5.py --finetune --experiment_name experiment \
  --eval_only --predict_test --num_beams 1 --max_new_tokens 512
```

### From-scratch training

Pass `--learning_rate 0.0001` explicitly; the diagnostic runner's default is
0.001. Use a fresh run directory for a new experiment:

```bash
python -u train_t5_scratch.py preflight --run_dir scratch_runs/short_lr1e4 --device cuda
python -u train_t5_scratch.py overfit --run_dir scratch_runs/short_lr1e4 \
  --device cuda --learning_rate 0.0001 --max_steps 600
python -u train_t5_scratch.py pilot --run_dir scratch_runs/short_lr1e4 \
  --device cuda --learning_rate 0.0001 --epochs 20
python -u scratch_autopilot.py --run_dir scratch_runs/short_lr1e4 \
  --max_epochs 60 --min_epochs 30 --patience 3 --min_delta 0.002
```

After the controller has stopped, export the selected model:

```bash
python -u export_scratch_test.py --run_dir scratch_runs/short_lr1e4 --device cuda
```

The export is saved under `scratch_runs/short_lr1e4/final/` and is protected
against overwrite. Generate a diagnostics archive with:

```bash
python train_t5_scratch.py bundle --run_dir scratch_runs/short_lr1e4
```

### SQL timeout recheck and development evaluation

The baseline executor uses a five-second limit. To retry timed-out queries with
a 30-second limit and write a new record file:

```bash
python recheck_timeouts.py \
  --sql results/t5_ft_experiment_dev.sql \
  --records records/t5_ft_experiment_dev.pkl \
  --output records/t5_ft_experiment_dev_rechecked.pkl \
  --timeout 30 --workers 2

python evaluate.py \
  --predicted_sql results/t5_ft_experiment_dev.sql \
  --predicted_records records/t5_ft_experiment_dev_rechecked.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

For other exports, pass their corresponding SQL and record paths plus a new
output path. The utility preserves the original SQL and record files.

## Checkpoint restoration

See [CHECKPOINTS.md](CHECKPOINTS.md) for the expected saved-model directories.

## Validation

Run from the repository root in the Part 2 environment:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The export checks require the T5 tokenizer to be cached or downloadable.
Set `HF_HOME` to reuse a local cache.
