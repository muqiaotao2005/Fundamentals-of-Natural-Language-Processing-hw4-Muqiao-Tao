"""Three gated, resumable diagnostics for randomly initialized T5-small."""
import argparse
import gc
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration, get_constant_schedule_with_warmup

from load_data import MODEL_NAME, get_tokenizer, normal_collate_fn
from utils import set_random_seeds
from scratch_diagnostics import (
    batch_check, execute_queries, fingerprint, generate, load_split, loader,
    loss_stats, masked_loss, read_reference, record_scores, save_predictions,
    validate_tokenizer, write_json,
)

BASE = Path(__file__).resolve().parent


def new_random_model(tokenizer, device):
    # Only load architecture metadata. Never load pretrained model weights.
    config = T5Config.from_pretrained(MODEL_NAME)
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.decoder_start_token_id = tokenizer.pad_token_id
    return T5ForConditionalGeneration(config).to(device)


def optimizer_and_scheduler(model, lr, warmup_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0,
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    return optimizer, scheduler


def update(model, optimizer, scheduler, batch, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, count, hits = masked_loss(model, batch, device)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return loss.item(), count, hits, float(norm)


def save_latest(path, model, optimizer, scheduler, sampler, position, history,
                settings, data_hash, best):
    state = dict(format_version=1, model=model.state_dict(),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 sampler=sampler.get_state(), position=position, history=history,
                 settings=settings, data_hash=data_hash, best=best,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_latest(path, model, optimizer, scheduler, sampler, settings, data_hash):
    # This is our own training-state checkpoint, not a third-party pickle.
    state = torch.load(path, map_location='cpu')
    if state['settings'] != settings or state['data_hash'] != data_hash:
        raise ValueError('Resume settings/data/code changed. Use a new run_dir for a new experiment.')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    sampler.set_state(state['sampler'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Resume requires the same device type as the saved run')
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['position'], state['history'], state['best']


def save_best(folder, model, tokenizer, metrics):
    model.save_pretrained(folder / 'best')
    tokenizer.save_pretrained(folder / 'best')
    write_json(folder / 'best_metrics.json', metrics)


def prepare_stage(args, tokenizer, data_hash, device):
    folder = args.run_dir / args.stage
    if args.resume:
        if not (folder / 'latest.pt').is_file():
            raise ValueError(f'No resumable checkpoint in {folder}')
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError(f'{folder} already contains outputs; use --resume or a new --run_dir')
    folder.mkdir(parents=True, exist_ok=True)
    set_random_seeds(args.seed)
    model = new_random_model(tokenizer, device)
    sampler = torch.Generator().manual_seed(args.seed)
    settings = dict(stage=args.stage, seed=args.seed, learning_rate=args.learning_rate,
                    batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                    warmup_steps=args.warmup_steps, max_new_tokens=args.max_new_tokens,
                    device_type=device.type, model_name=MODEL_NAME,
                    tiny_examples=args.tiny_examples, tiny_max_target_tokens=args.tiny_max_target_tokens,
                    eval_every_steps=args.eval_every_steps, gate_sql_em=args.gate_sql_em,
                    probe_size=args.probe_size, full_eval_every=args.full_eval_every,
                    sql_timeout=args.sql_timeout, sql_workers=args.sql_workers)
    optimizer, scheduler = optimizer_and_scheduler(model, args.learning_rate, args.warmup_steps)
    position, history, best = 0, [], {'score': -1.0}
    if args.resume:
        position, history, best = restore_latest(folder / 'latest.pt', model, optimizer,
                                                scheduler, sampler, settings, data_hash)
    write_json(folder / 'run_config.json', dict(settings, data_hash=data_hash,
                initialization='all learned weights random; config only from google-t5/t5-small',
                max_steps=args.max_steps, epochs=args.epochs, resume=args.resume))
    return folder, model, sampler, settings, optimizer, scheduler, position, history, best


def require_gate(args, name, data_hash):
    path = args.run_dir / name / 'summary.json'
    if not path.is_file():
        raise ValueError(f'Run the {name} stage first: {path} is missing')
    summary = json.loads(path.read_text())
    if not summary.get('passed') or summary.get('data_hash') != data_hash:
        raise ValueError(f'{name} gate is not passed for the current data/code/tokenizer')


def preflight(args, tokenizer, train, train_q, train_sql, dev, dev_q, dev_sql, data_hash, device):
    folder = args.run_dir / 'preflight'
    report = dict(passed=False, data_hash=data_hash, device=str(device))
    write_json(folder / 'summary.json', report)
    started = time.monotonic()
    report['train'] = validate_tokenizer(tokenizer, train_q, train_sql, args.max_new_tokens)
    report['dev'] = validate_tokenizer(tokenizer, dev_q, dev_sql, args.max_new_tokens)
    set_random_seeds(args.seed)
    model = new_random_model(tokenizer, device)
    shortest = sorted(range(len(train)), key=lambda i: len(train[i][1]))[:2]
    batch = normal_collate_fn([train[i] for i in shortest])
    report['model'] = dict(parameters=sum(p.numel() for p in model.parameters()),
                           encoder_layers=model.config.num_layers,
                           decoder_layers=model.config.num_decoder_layers,
                           vocab_size=len(tokenizer), initialization='random')
    report['batch_check'] = batch_check(model, batch, tokenizer, device)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    reference = read_reference(args.data_dir)
    if len(reference) != len(dev):
        raise ValueError('Reference records do not align with dev questions')
    selected = random.Random(args.seed).sample(range(len(dev)), min(16, len(dev)))
    # Regression cases from the observed five-second timeout issue, zero-based.
    if len(dev) == 466:
        selected += [40, 187, 191, 247]
    selected = sorted(set(selected))
    records, errors, seconds = execute_queries(args.data_dir / 'flight_database.db',
            [dev_sql[i] for i in selected], args.sql_timeout, args.sql_workers)
    report['gold_execution'] = [dict(line=i+1, error=e, seconds=t,
                matches_reference=set(r) == set(reference[i]))
                for i, r, e, t in zip(selected, records, errors, seconds)]
    report['passed'] = all(not row['error'] and row['matches_reference']
                           for row in report['gold_execution'])
    report['seconds'] = time.monotonic()-started
    write_json(folder / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise ValueError('Gold execution check failed; inspect preflight/summary.json before training')


def overfit(args, tokenizer, train, questions, gold_sql, data_hash, device):
    require_gate(args, 'preflight', data_hash)
    eligible = [i for i in range(len(train)) if len(train[i][1]) <= args.tiny_max_target_tokens]
    # Avoid contradictory duplicate questions in a memorization diagnostic.
    unique = {}
    for i in eligible:
        unique.setdefault(questions[i], i)
    if len(unique) < args.tiny_examples:
        raise ValueError('Not enough distinct short training examples for the overfit check')
    indices = sorted(random.Random(args.seed).sample(list(unique.values()), args.tiny_examples))
    if max(len(train[i][1]) for i in indices) > args.max_new_tokens:
        raise ValueError('Overfit generation limit cannot represent the selected SQL targets')
    stage = prepare_stage(args, tokenizer, data_hash, device)
    folder, model, sampler, settings, optimizer, scheduler, position, history, best = stage
    if position > args.max_steps:
        raise ValueError('Saved update is beyond --max_steps; do not shrink the resume budget')
    write_json(folder / 'selected_examples.json', [dict(source_line=i+1, question=questions[i],
                                                      sql=gold_sql[i]) for i in indices])
    eval_loader = loader(train, indices, args.eval_batch_size)
    if not args.resume:
        initial = loss_stats(model, eval_loader, device)
        write_json(folder / 'initial_metrics.json', initial)
        save_latest(folder/'latest.pt', model, optimizer, scheduler, sampler, 0, history,
                    settings, data_hash, best)
    passed = best['score'] >= args.gate_sql_em
    started = time.monotonic()
    last_grad = None
    for step in range(position+1, args.max_steps+1):
        if passed:
            break
        chosen = torch.randperm(len(indices), generator=sampler)[:args.batch_size].tolist()
        batch = normal_collate_fn([train[indices[i]] for i in chosen])
        train_loss, _, _, last_grad = update(model, optimizer, scheduler, batch, device)
        position = step
        if step % 25 == 0:
            print(f'Overfit update {step}/{args.max_steps}: loss={train_loss:.4f}', flush=True)
        if step % args.eval_every_steps != 0 and step != args.max_steps:
            continue
        stats = loss_stats(model, eval_loader, device)
        queries, diag, generation_stats = generate(model, eval_loader, tokenizer, device, args.max_new_tokens)
        exact = sum(q == gold_sql[i] for q, i in zip(queries, indices))/len(indices)
        row = dict(step=step, sql_em=exact, gradient_norm=last_grad,
                   learning_rate=optimizer.param_groups[0]['lr'], **stats, **generation_stats)
        history.append(row)
        save_predictions(folder / f'predictions/step_{step:04d}', indices, questions, gold_sql, queries, diag)
        if exact > best['score']:
            best = dict(score=exact, step=step)
            save_best(folder, model, tokenizer, row)
        passed = exact >= args.gate_sql_em
        write_json(folder / 'history.json', history)
        save_latest(folder/'latest.pt', model, optimizer, scheduler, sampler, step, history,
                    settings, data_hash, best)
        print(json.dumps(row), flush=True)
    result = dict(passed=passed, stage='overfit', data_hash=data_hash,
                  completed_updates=position, best_sql_em=best['score'], threshold=args.gate_sql_em,
                  best_step=best.get('step'), examples=len(indices),
                  this_invocation_seconds=time.monotonic()-started,
                  next_action='Run a fresh full-data pilot' if passed else
                  'Inspect loss and predictions; diagnose or explicitly extend this step budget')
    write_json(folder / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return passed


def evaluate_queries(args, model, tokenizer, dev, questions, gold_sql, reference,
                     indices, prefix, device):
    batches = loader(dev, indices, args.eval_batch_size)
    queries, diag, generation_stats = generate(model, batches, tokenizer, device, args.max_new_tokens)
    records, errors, seconds = execute_queries(args.data_dir/'flight_database.db', queries,
                                               args.sql_timeout, args.sql_workers)
    scores = record_scores([gold_sql[i] for i in indices], queries,
                           [reference[i] for i in indices], records, errors)
    save_predictions(prefix, indices, questions, gold_sql, queries, diag, records, errors)
    return dict(**scores, **generation_stats, evaluation_examples=len(indices),
                max_query_seconds=max(seconds))


def pilot(args, tokenizer, train, dev, questions, gold_sql, data_hash, device):
    require_gate(args, 'preflight', data_hash)
    require_gate(args, 'overfit', data_hash)
    stage = prepare_stage(args, tokenizer, data_hash, device)
    folder, model, sampler, settings, optimizer, scheduler, position, history, best = stage
    if position > args.epochs:
        raise ValueError('Saved epoch is beyond --epochs; do not shrink the resume budget')
    reference = read_reference(args.data_dir)
    if len(reference) != len(dev):
        raise ValueError('Dev reference count mismatch')
    probe = sorted(random.Random(args.seed).sample(range(len(dev)), min(args.probe_size, len(dev))))
    write_json(folder / 'probe_lines.json', [i+1 for i in probe])
    train_loader = loader(train, range(len(train)), args.batch_size, True, sampler)
    dev_loader = loader(dev, range(len(dev)), args.eval_batch_size)
    if not args.resume:
        write_json(folder / 'initial_metrics.json', loss_stats(model, dev_loader, device))
        save_latest(folder/'latest.pt', model, optimizer, scheduler, sampler, 0, history,
                    settings, data_hash, best)
    started = time.monotonic()
    for epoch in range(position+1, args.epochs+1):
        epoch_start = time.monotonic()
        total_loss, total_tokens, hits, max_norm = 0.0, 0, 0, 0.0
        for batch in tqdm(train_loader, desc=f'Pilot epoch {epoch}/{args.epochs}'):
            loss, n, correct, norm = update(model, optimizer, scheduler, batch, device)
            total_loss += loss*n
            total_tokens += n
            hits += correct
            max_norm = max(max_norm, norm)
        train_seconds = time.monotonic()-epoch_start
        dev_stats = loss_stats(model, dev_loader, device)
        full = epoch % args.full_eval_every == 0 or epoch == args.epochs
        indices = list(range(len(dev))) if full else probe
        scope = 'dev_full' if full else 'dev_probe'
        evaluation = evaluate_queries(args, model, tokenizer, dev, questions, gold_sql,
                         reference, indices, folder/f'predictions/epoch_{epoch:03d}_{scope}', device)
        row = dict(epoch=epoch, train_loss=total_loss/total_tokens,
                   train_token_accuracy=hits/total_tokens, dev_loss=dev_stats['loss'],
                   dev_token_accuracy=dev_stats['token_accuracy'], evaluation_scope=scope,
                   learning_rate=optimizer.param_groups[0]['lr'], max_gradient_norm=max_norm,
                   train_seconds=train_seconds, **evaluation)
        if full and row['record_f1'] > best['score']:
            best = dict(score=row['record_f1'], epoch=epoch)
            save_best(folder, model, tokenizer, row)
        row['epoch_seconds_before_latest_save'] = time.monotonic()-epoch_start
        history.append(row)
        position = epoch
        write_json(folder / 'history.json', history)
        save_latest(folder/'latest.pt', model, optimizer, scheduler, sampler, epoch, history,
                    settings, data_hash, best)
        print(json.dumps(row), flush=True)
    summary = dict(stage='pilot', status='completed', data_hash=data_hash, completed_epochs=position,
                   best_full_dev_record_f1=best['score'], best_epoch=best.get('epoch'),
                   this_invocation_seconds=time.monotonic()-started, test_used=False,
                   next_action='Review full-dev metrics, loss curves and generated SQL before a long run')
    write_json(folder / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def bundle_results(run_dir):
    run_dir = Path(run_dir)
    target = run_dir.with_name(run_dir.name+'_diagnostics.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        if run_dir.is_dir():
            for p in sorted(run_dir.rglob('*')):
                relative = p.relative_to(run_dir)
                if p.is_file() and not p.is_symlink() and 'best' not in relative.parts:
                    if p.suffix in {'.json', '.jsonl', '.sql', '.pkl', '.log', '.txt'}:
                        archive.write(p, relative.as_posix())
    print(f'Diagnostics bundle: {target}', flush=True)
    return target


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['preflight', 'overfit', 'pilot', 'bundle'])
    p.add_argument('--run_dir', type=Path, default=BASE/'scratch_runs/short_v1')
    p.add_argument('--data_dir', type=Path, default=BASE/'data')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=16)
    p.add_argument('--warmup_steps', type=int)
    p.add_argument('--max_steps', type=int, default=600)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--tiny_examples', type=int, default=16)
    p.add_argument('--tiny_max_target_tokens', type=int, default=128)
    p.add_argument('--eval_every_steps', type=int, default=100)
    p.add_argument('--gate_sql_em', type=float, default=.9)
    p.add_argument('--probe_size', type=int, default=64)
    p.add_argument('--full_eval_every', type=int, default=5)
    p.add_argument('--max_new_tokens', type=int)
    p.add_argument('--sql_timeout', type=float, default=30)
    p.add_argument('--sql_workers', type=int, default=2)
    args = p.parse_args()
    for name in ('batch_size', 'eval_batch_size', 'max_steps', 'epochs', 'tiny_examples',
                 'tiny_max_target_tokens', 'eval_every_steps', 'probe_size', 'full_eval_every', 'sql_workers'):
        if getattr(args, name) <= 0:
            p.error(f'--{name} must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error('--learning_rate must be finite and positive')
    if not math.isfinite(args.sql_timeout) or args.sql_timeout <= 0:
        p.error('--sql_timeout must be finite and positive')
    if not 0 < args.gate_sql_em <= 1:
        p.error('--gate_sql_em must be in (0, 1]')
    if args.warmup_steps is not None and args.warmup_steps < 0:
        p.error('--warmup_steps must be nonnegative')
    if args.max_new_tokens is None:
        args.max_new_tokens = 160 if args.stage == 'overfit' else 512
    if args.max_new_tokens < 1:
        p.error('--max_new_tokens must be positive')
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def main():
    args = parse_args()
    if args.stage == 'bundle':
        bundle_results(args.run_dir)
        return 0
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = get_tokenizer()
    train, train_q, train_sql = load_split(args.data_dir, 'train')
    dev, dev_q, dev_sql = load_split(args.data_dir, 'dev')
    data_hash = fingerprint(args.data_dir, tokenizer)
    if args.warmup_steps is None:
        args.warmup_steps = math.ceil(len(train)/args.batch_size) if args.stage == 'pilot' else 25
    if args.stage == 'overfit' and args.warmup_steps >= args.max_steps:
        raise ValueError('Overfit budget must extend beyond warmup')
    if args.stage == 'pilot' and args.warmup_steps >= args.epochs*math.ceil(len(train)/args.batch_size):
        raise ValueError('Pilot budget must extend beyond warmup')
    print(f'Stage={args.stage}; device={device}; train={len(train)}; dev={len(dev)}; '
          f'outputs={args.run_dir}; no test data loaded', flush=True)
    try:
        if args.stage == 'preflight':
            preflight(args, tokenizer, train, train_q, train_sql, dev, dev_q, dev_sql, data_hash, device)
        elif args.stage == 'overfit':
            return 0 if overfit(args, tokenizer, train, train_q, train_sql, data_hash, device) else 2
        else:
            pilot(args, tokenizer, train, dev, dev_q, dev_sql, data_hash, device)
    finally:
        bundle_results(args.run_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
