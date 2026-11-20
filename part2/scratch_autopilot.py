"""Resume an existing scratch pilot in bounded chunks, with full-dev early stopping.

This controller deliberately leaves the fingerprinted training code unchanged.
Only load latest.pt from your own completed/ongoing scratch experiment.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
TRAINING_OPTIONS = (
    'seed', 'learning_rate', 'batch_size', 'eval_batch_size', 'warmup_steps',
    'max_new_tokens', 'tiny_examples', 'tiny_max_target_tokens',
    'eval_every_steps', 'gate_sql_em', 'probe_size', 'full_eval_every',
    'sql_timeout', 'sql_workers',
)


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def active_pilots(run_dir, proc_root=Path('/proc')):
    """Wait for the user's already running pilot before reading its checkpoint."""
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = process.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            script = next((i for i, arg in enumerate(argv)
                           if Path(arg).name == 'train_t5_scratch.py'), None)
            if script is None or argv[script+1:script+2] != ['pilot']:
                continue
            value = next((arg.split('=', 1)[1] for arg in argv if arg.startswith('--run_dir=')), None)
            if '--run_dir' in argv:
                value = argv[argv.index('--run_dir') + 1]
            cwd = process.joinpath('cwd').resolve(strict=True)
            if value is None:
                value = str((cwd / argv[script]).resolve().parent / 'scratch_runs/short_v1')
            target = Path(value)
            if not target.is_absolute():
                target = cwd / target
            if target.resolve() == run_dir.resolve():
                found.append(int(process.name))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue  # Processes can finish while /proc is being inspected.
    return found


def read_checkpoint(run_dir):
    import torch
    # Trusted checkpoint produced by our own runner, including optimizer/RNG.
    checkpoint = torch.load(run_dir / 'pilot/latest.pt', map_location='cpu', weights_only=False)
    state = {key: checkpoint[key] for key in ('settings', 'data_hash', 'position', 'history', 'best')}
    del checkpoint
    if state['settings']['stage'] != 'pilot':
        raise ValueError('Expected a pilot checkpoint')
    if [row['epoch'] for row in state['history']] != list(range(1, state['position'] + 1)):
        raise ValueError('Checkpoint history does not match its committed epoch')
    for stage in ('preflight', 'overfit'):
        gate = json.loads((run_dir / stage / 'summary.json').read_text())
        if not gate.get('passed') or gate.get('data_hash') != state['data_hash']:
            raise ValueError(f'{stage} gate is not passed for this checkpoint')
    return state


def stopping_progress(history, min_delta):
    anchor, stale = None, 0
    for row in history:
        if row['evaluation_scope'] != 'dev_full':
            continue
        score = float(row['record_f1'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid full-dev F1; stop and inspect the evaluation')
        if anchor is None or score > anchor + min_delta:
            anchor, stale = score, 0
        else:
            stale += 1
    return dict(significant_best_f1=anchor, consecutive_full_evals_without_improvement=stale)


def next_epoch(current, interval, maximum):
    return min((current // interval + 1) * interval, maximum)


def training_command(run_dir, state, target):
    settings = state['settings']
    command = [sys.executable, '-u', str(BASE / 'train_t5_scratch.py'), 'pilot',
               '--run_dir', str(run_dir), '--data_dir', str(BASE / 'data'),
               '--device', settings['device_type'], '--resume', '--epochs', str(target)]
    for name in TRAINING_OPTIONS:
        command.extend(['--' + name, str(settings[name])])
    return command


def run_training(command):
    subprocess.run(command, cwd=BASE, check=True)


def continue_training(args, read_state=read_checkpoint, run_child=run_training):
    state = read_state(args.run_dir)
    started_epoch = state['position']
    settings = state['settings'].copy()
    data_hash = state['data_hash']
    started = time.monotonic()
    while True:
        progress = stopping_progress(state['history'], args.min_delta)
        common = dict(controller_pid=os.getpid(), started_epoch=started_epoch,
                      completed_epochs=state['position'], max_epochs=args.max_epochs,
                      min_epochs=args.min_epochs, patience_full_evals=args.patience,
                      min_delta=args.min_delta, learning_rate=settings['learning_rate'],
                      best_epoch=state['best'].get('epoch'), best_full_dev_record_f1=state['best']['score'],
                      best_model_path=str(args.run_dir / 'pilot/best'), test_used=False,
                      elapsed_seconds=time.monotonic() - started, **progress)
        if state['position'] >= args.max_epochs:
            reason = 'max_epochs_reached'
        elif (state['position'] >= args.min_epochs and
              progress['consecutive_full_evals_without_improvement'] >= args.patience):
            reason = 'full_dev_plateau'
        else:
            reason = None
        if reason:
            summary = dict(common, status='stopped', stop_reason=reason)
            write_json(args.run_dir / 'autopilot_summary.json', summary)
            write_json(args.run_dir / 'autopilot_status.json', summary)
            print(json.dumps(summary, indent=2), flush=True)
            return summary
        target = next_epoch(state['position'], settings['full_eval_every'], args.max_epochs)
        write_json(args.run_dir / 'autopilot_status.json', dict(common, status='training', target_epoch=target))
        print(f'Autopilot: resume after epoch {state["position"]}; train through {target}.', flush=True)
        run_child(training_command(args.run_dir, state, target))
        updated = read_state(args.run_dir)
        if updated['position'] != target:
            raise ValueError('Child did not commit the expected epoch; refusing to repeat training')
        if updated['settings'] != settings or updated['data_hash'] != data_hash:
            raise ValueError('Checkpoint settings or data changed during this continuation')
        state = updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, default=BASE / 'scratch_runs/short_lr1e4')
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--min_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=3, help='Consecutive full-dev checks without significant gain')
    parser.add_argument('--min_delta', type=float, default=.002, help='F1 fraction: .002 = 0.2 percentage points')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if not 0 <= args.min_epochs <= args.max_epochs or args.max_epochs < 1 or args.patience < 1:
        parser.error('Require 0 <= min_epochs <= max_epochs and positive max_epochs/patience')
    if not math.isfinite(args.min_delta) or not 0 <= args.min_delta < 1:
        parser.error('min_delta must be finite and in [0, 1)')
    if not (args.run_dir / 'pilot/latest.pt').is_file():
        parser.error('An existing pilot/latest.pt is required; this controller never starts from scratch')
    # Only one controller may own this run, even after browser disconnection.
    with (args.run_dir / '.autopilot.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('An autopilot already owns this run. Check autopilot.log.', flush=True)
            return 2
        try:
            while True:
                running = active_pilots(args.run_dir)
                if not running:
                    break
                print(f'Autopilot: waiting for existing pilot process(es) {running}.', flush=True)
                write_json(args.run_dir / 'autopilot_status.json',
                           dict(status='waiting', controller_pid=os.getpid(), waiting_for_pids=running))
                time.sleep(20)
            continue_training(args)
            # Include the controller's final report in the existing diagnostic ZIP.
            subprocess.run([sys.executable, str(BASE / 'train_t5_scratch.py'), 'bundle',
                            '--run_dir', str(args.run_dir)], cwd=BASE, check=True)
        except (Exception, KeyboardInterrupt) as error:
            write_json(args.run_dir / 'autopilot_status.json',
                       dict(status='error', controller_pid=os.getpid(), error=str(error)))
            print(f'Autopilot stopped on error: {error}', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
