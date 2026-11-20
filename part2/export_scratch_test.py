"""Export the dev-selected scratch model's extra-credit submission, without training."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
import zipfile

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from load_data import INPUT_PREFIX, load_lines, test_collate_fn
from scratch_autopilot import active_pilots
from scratch_diagnostics import (error_kind, execute_queries, fingerprint, generate,
                                 read_reference, record_scores, write_json)
from utils import set_random_seeds

BASE = Path(__file__).resolve().parent
DEV_EXAMPLES, TEST_EXAMPLES = 466, 432
SUBMISSION_STEM = 't5_ft_experiment_ec_test'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_run(run_dir):
    pilot = run_dir / 'pilot'
    config = json.loads((pilot / 'run_config.json').read_text())
    history = json.loads((pilot / 'history.json').read_text())
    metrics = json.loads((pilot / 'best_metrics.json').read_text())
    stopped = json.loads((run_dir / 'autopilot_summary.json').read_text())
    status = json.loads((run_dir / 'autopilot_status.json').read_text())
    if stopped['status'] != 'stopped' or status['status'] != 'stopped':
        raise ValueError('Wait for autopilot to finish before exporting the selected model')
    if config['stage'] != 'pilot' or not config['initialization'].startswith('all learned weights random;'):
        raise ValueError('Expected the randomly initialized scratch pilot')
    full = [row for row in history if row['evaluation_scope'] == 'dev_full']
    if not full:
        raise ValueError('No full-development model selection results')
    chosen = max(full, key=lambda row: row['record_f1'])
    if (chosen['epoch'] != metrics['epoch'] or chosen['record_f1'] != metrics['record_f1'] or
            chosen['epoch'] != stopped['best_epoch'] or
            chosen['record_f1'] != stopped['best_full_dev_record_f1']):
        raise ValueError('Best model metadata does not match full-dev selection history')
    if history[-1]['epoch'] != stopped['completed_epochs']:
        raise ValueError('Training history changed after autopilot stopped')
    return config, metrics


def load_saved_model(model_dir, device):
    tokenizer = T5TokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).to(device)
    config = model.config
    if (config.d_model, config.num_layers, config.num_decoder_layers, config.num_heads) != (512, 6, 6, 8):
        raise ValueError('Expected the T5-small architecture')
    if (config.pad_token_id, config.eos_token_id, config.decoder_start_token_id) != (0, 1, 0):
        raise ValueError('Unexpected generation special tokens')
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise ValueError('Saved tokenizer and model vocabulary differ')
    return model.eval(), tokenizer


def inference_collate(batch):
    source, mask, initial = test_collate_fn(batch)
    # Reuse the same generation helper as validation; no target tokens are passed.
    return source, mask, None, None, initial


def question_loader(questions, tokenizer, batch_size):
    encoded = tokenizer([INPUT_PREFIX + question for question in questions],
                        truncation=False)['input_ids']
    dataset = [(torch.tensor(ids, dtype=torch.long),) for ids in encoded]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=inference_collate)


def save_outputs(folder, stem, questions, queries, diagnostics, records, errors, gold_sql=None):
    n = len(questions)
    if not all(len(values) == n for values in (queries, diagnostics, records, errors)):
        raise ValueError('Predictions/records do not align with input questions')
    if gold_sql is not None and len(gold_sql) != n:
        raise ValueError('Development SQL count mismatch')
    (folder / (stem + '.sql')).write_text(''.join(query + '\n' for query in queries))
    with (folder / (stem + '.pkl')).open('wb') as output:
        pickle.dump((records, errors), output)
    with (folder / (stem + '.jsonl')).open('w') as output:
        for i, question in enumerate(questions):
            row = dict(source_line=i+1, question=question, predicted_sql=queries[i],
                       record_count=len(set(records[i])), execution_error=errors[i], **diagnostics[i])
            if gold_sql is not None:
                row['gold_sql'] = gold_sql[i]
            output.write(json.dumps(row, ensure_ascii=False) + '\n')


def export(args):
    started = time.monotonic()
    destination = args.run_dir / 'final'
    if destination.exists():
        raise ValueError(f'{destination} already exists; inspect the saved outputs before exporting again')
    config, best = selected_run(args.run_dir)
    model_dir = args.run_dir / 'pilot/best'
    set_random_seeds(config['seed'])
    model, tokenizer = load_saved_model(model_dir, args.device)
    if fingerprint(args.data_dir, tokenizer) != config['data_hash']:
        raise ValueError('Training data/code/tokenizer fingerprint changed')
    batch_size = config['eval_batch_size']
    max_tokens = config['max_new_tokens']
    sql_timeout, sql_workers = config['sql_timeout'], config['sql_workers']
    weights = model_dir / 'model.safetensors'
    if not weights.is_file():
        weights = model_dir / 'pytorch_model.bin'
    weights_hash = sha256(weights)
    work = Path(tempfile.mkdtemp(prefix='final_in_progress_', dir=args.run_dir))
    print(f'Loading saved best epoch {best["epoch"]}; full-dev F1 {best["record_f1"]:.4%}.', flush=True)
    print(f'Inference only; temporary outputs: {work}', flush=True)

    dev_questions = load_lines(args.data_dir / 'dev.nl')
    dev_gold = load_lines(args.data_dir / 'dev.sql')
    reference = read_reference(args.data_dir)
    if not len(dev_questions) == len(dev_gold) == len(reference) == DEV_EXAMPLES:
        raise ValueError('Unexpected development data size or ordering')
    dev_sql, dev_diag, dev_generation = generate(model,
        question_loader(dev_questions, tokenizer, batch_size), tokenizer, args.device, max_tokens)
    original = load_lines(args.run_dir / f'pilot/predictions/epoch_{best["epoch"]:03d}_dev_full.sql')
    if dev_sql != original:
        (work / 'dev_reproduced.sql').write_text(''.join(query+'\n' for query in dev_sql))
        raise ValueError(f'Best-model dev SQL differs from saved predictions; test was not generated. Inspect {work}')
    dev_records, dev_errors, _ = execute_queries(args.data_dir / 'flight_database.db',
                                                dev_sql, sql_timeout, sql_workers)
    dev_metrics = record_scores(dev_gold, dev_sql, reference, dev_records, dev_errors)
    save_outputs(work, 'dev', dev_questions, dev_sql, dev_diag, dev_records, dev_errors, dev_gold)
    print('Dev SQL reproduction: exact match with saved best-epoch predictions.', flush=True)
    print('Rechecked full dev: ' + json.dumps(dev_metrics), flush=True)

    # Read only the unlabeled test questions, after dev selection and verification.
    test_questions = load_lines(args.data_dir / 'test.nl')
    if len(test_questions) != TEST_EXAMPLES or not all(test_questions):
        raise ValueError('Expected 432 nonempty test questions in the original file order')
    test_sql, test_diag, test_generation = generate(model,
        question_loader(test_questions, tokenizer, batch_size), tokenizer, args.device, max_tokens)
    test_records, test_errors, _ = execute_queries(args.data_dir / 'flight_database.db',
                                                  test_sql, sql_timeout, sql_workers)
    save_outputs(work, SUBMISSION_STEM, test_questions, test_sql, test_diag, test_records, test_errors)
    summary = dict(selected_epoch=best['epoch'], model_path=str(model_dir), weights_sha256=weights_hash,
                   selection_metric='Record F1 on the full development set',
                   saved_best_dev_metrics=best, dev_sql_reproduced_exactly=True,
                   rechecked_dev_metrics=dev_metrics, dev_generation=dev_generation,
                   test_examples=len(test_questions), test_labels_used=False,
                   test_record_f1=None, test_record_em=None, test_sql_em=None,
                   test_execution_error_count=sum(bool(e) for e in test_errors),
                   test_execution_error_counts=dict(Counter(error_kind(e) for e in test_errors)),
                   test_generation=test_generation, test_input_sha256=sha256(args.data_dir / 'test.nl'),
                   generation=dict(num_beams=1, do_sample=False, max_new_tokens=max_tokens, batch_size=batch_size),
                   sql_execution=dict(timeout_seconds=sql_timeout, workers=sql_workers),
                   elapsed_seconds=time.monotonic()-started,
                   note='Hidden test scores are unavailable locally. SQL runtime can affect record-based metrics.')
    names = [SUBMISSION_STEM + '.sql', SUBMISSION_STEM + '.pkl']
    summary['submission_files'] = {name: dict(sha256=sha256(work/name), bytes=(work/name).stat().st_size)
                                   for name in names}
    write_json(work / 'summary.json', summary)
    with zipfile.ZipFile(work / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(work / name, name)
    work.rename(destination)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved {len(test_sql)} test SQL queries and aligned records.', flush=True)
    print(f'Extra-credit submission files: {destination / "submission.zip"}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, default=BASE / 'scratch_runs/short_lr1e4')
    parser.add_argument('--data_dir', type=Path, default=BASE / 'data')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    args.run_dir, args.data_dir = args.run_dir.resolve(), args.data_dir.resolve()
    args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    with (args.run_dir / '.autopilot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active_pilots(args.run_dir):
            raise RuntimeError('A pilot is still running; wait for it to finish')
        export(args)


if __name__ == '__main__':
    main()
