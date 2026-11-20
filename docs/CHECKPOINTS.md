# Restoring the submitted checkpoints

This repository contains source code, not trained weights. Download the
selected checkpoints from the Google Drive folder linked in the written
report. Diagnostics and prediction ZIP files are not model checkpoints.

## Fine-tuned T5 (selected epoch 18)

Restore the saved experiment under `part2/checkpoints/ft_experiments/experiment/`.
Keep its `best/` folder, tokenizer files at the experiment root, `run_config.json`,
and `history.json`. The `best/` folder must contain the original `config.json`
and trained weights (`model.safetensors` or `pytorch_model.bin`), plus any other
files produced by `save_pretrained`.

Activate the Part 2 environment and run from `part2/`:

```bash
python train_t5.py --finetune --experiment_name experiment \
  --eval_only --predict_test --num_beams 1 --max_new_tokens 512
```

The baseline reconstructs the modified tokenizer from `google-t5/t5-small`, so
the base tokenizer must be downloadable or cached. Saved tokenizer files
document the vocabulary used. The SQL executor uses a five-second limit; see
[RUNNING.md](RUNNING.md) for execution-only timeout rechecks.

## Randomly initialized T5 (selected epoch 35)

Restore the experiment under `part2/scratch_runs/short_lr1e4/`, including:

```text
pilot/best/                         # all model, config, and tokenizer files
pilot/best_metrics.json
pilot/run_config.json
pilot/history.json
pilot/predictions/                  # saved selected-epoch dev SQL / records
autopilot_summary.json
autopilot_status.json
```

Keeping the entire experiment directory is easiest. `pilot/latest.pt` is needed
to resume training; it is not the selected epoch-35 model. Training-state files
should come from your own trusted run.

For a fresh export, the `final/` directory must not already contain an export;
the script protects existing outputs from overwrite:

```bash
python export_scratch_test.py --run_dir scratch_runs/short_lr1e4 --device cuda
```

The exporter checks selection metadata and reproduces saved best-epoch dev SQL
before reading test questions. Restoring model weights alone is sufficient for
loading with Transformers, but this guarded export also needs the metadata and
dev predictions listed above.

## Part 1 (optional)

Part 1 inference expects the saved BERT model in `part1/out/` or
`part1/out_augmented/`. The tokenizer is `bert-base-cased`.
