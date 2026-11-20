"""Controller checks: existing-checkpoint resume, bounds, early stopping, failures."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
PART2 = ROOT / 'part2'
spec = importlib.util.spec_from_file_location('autopilot', PART2 / 'scratch_autopilot.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)
OPTIONS = dict(stage='pilot', seed=42, learning_rate=.0001, batch_size=16,
               eval_batch_size=16, warmup_steps=265, max_new_tokens=512,
               device_type='cuda', model_name='google-t5/t5-small', tiny_examples=16,
               tiny_max_target_tokens=128, eval_every_steps=100, gate_sql_em=.9,
               probe_size=64, full_eval_every=5, sql_timeout=30, sql_workers=2)


def row(epoch, score, scope='dev_full'):
    return dict(epoch=epoch, record_f1=score, evaluation_scope=scope)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(run_dir=self.root, max_epochs=60, min_epochs=30,
                                    patience=3, min_delta=.002)
        self.state = dict(settings=OPTIONS.copy(), data_hash='fixture', position=20,
                          history=[row(5,.23),row(10,.41),row(15,.49),row(20,.565)],
                          best=dict(epoch=20,score=.565))
        self.targets = []

    def tearDown(self):
        self.temp.cleanup()

    def reader(self, root):
        return copy.deepcopy(self.state)

    def child(self, command, scores):
        self.assertIn('--resume', command)
        self.assertNotIn('preflight', command)
        self.assertNotIn('overfit', command)
        self.assertEqual(command[command.index('--learning_rate')+1], '0.0001')
        self.assertEqual(command[command.index('--warmup_steps')+1], '265')
        target = int(command[command.index('--epochs')+1])
        self.targets.append(target)
        score = scores.get(target, .6)
        self.state['position'] = target
        self.state['history'].append(row(target,score))
        if score > self.state['best']['score']:
            self.state['best'] = dict(epoch=target,score=score)

    def test_plateau_stops_after_three_full_checks_and_keeps_raw_best(self):
        scores={25:.60,30:.599,35:.601,40:.599}
        summary=auto.continue_training(self.args,self.reader,lambda cmd:self.child(cmd,scores))
        self.assertEqual(self.targets,[25,30,35,40])
        self.assertEqual(summary['stop_reason'],'full_dev_plateau')
        self.assertEqual(summary['best_epoch'],35)
        self.assertEqual(summary['best_full_dev_record_f1'],.601)

    def test_improving_run_stops_at_total_epoch_cap(self):
        scores={epoch:.565+(epoch-20)*.002 for epoch in range(25,61,5)}
        summary=auto.continue_training(self.args,self.reader,lambda cmd:self.child(cmd,scores))
        self.assertEqual(self.targets,list(range(25,61,5)))
        self.assertEqual(summary['completed_epochs'],60)
        self.assertEqual(summary['stop_reason'],'max_epochs_reached')

    def test_probe_scores_do_not_change_patience_and_small_gains_accumulate(self):
        hist=[row(5,.5),row(6,.99,'dev_probe'),row(10,.501),row(15,.503)]
        result=auto.stopping_progress(hist,.002)
        self.assertEqual(result['consecutive_full_evals_without_improvement'],0)
        self.assertEqual(result['significant_best_f1'],.503)
        with self.assertRaises(ValueError):
            auto.stopping_progress([row(5,math.nan)],.002)

    def test_failure_does_not_trigger_retries_or_advance_checkpoint(self):
        calls=[]
        def fail(cmd):
            calls.append(cmd)
            raise subprocess.CalledProcessError(2,cmd)
        with self.assertRaises(subprocess.CalledProcessError):
            auto.continue_training(self.args,self.reader,fail)
        self.assertEqual(len(calls),1)
        self.assertEqual(self.state['position'],20)

    def test_changed_checkpoint_or_missing_progress_fails_closed(self):
        def mutate(cmd):
            self.child(cmd,{})
            self.state['settings']['learning_rate']=.001
        with self.assertRaises(ValueError):
            auto.continue_training(self.args,self.reader,mutate)
        self.state['settings']['learning_rate']=.0001
        with self.assertRaises(ValueError):
            auto.continue_training(self.args,self.reader,lambda cmd:None)

    def test_partial_chunk_resumes_at_next_full_eval_and_existing_cap_does_no_work(self):
        self.state['position']=23
        self.args.max_epochs=25
        self.args.min_epochs=25
        summary=auto.continue_training(self.args,self.reader,lambda cmd:self.child(cmd,{}))
        self.assertEqual(self.targets,[25])
        self.assertEqual(summary['completed_epochs'],25)
        auto.continue_training(self.args,self.reader,lambda cmd:self.fail('repeated training'))

    def test_all_training_settings_are_forwarded_without_defaults(self):
        command=auto.training_command(self.root,self.state,25)
        for key in auto.TRAINING_OPTIONS:
            self.assertEqual(command[command.index('--'+key)+1],str(OPTIONS[key]))

    def test_existing_manual_pilot_is_detected_but_other_run_is_not(self):
        proc=self.root/'proc'
        proc.mkdir()
        for pid, run in [('12345',self.root),('23456',self.root/'different')]:
            process=proc/pid
            process.mkdir()
            (process/'cwd').symlink_to(PART2,target_is_directory=True)
            (process/'cmdline').write_bytes(('python\0-u\0train_t5_scratch.py\0pilot\0--run_dir\0'+str(run)+'\0').encode())
        self.assertEqual(auto.active_pilots(self.root,proc),[12345])

    def test_reads_committed_checkpoint_history_not_uncommitted_json(self):
        import torch
        (self.root/'pilot').mkdir()
        hist=[row(i,.5,'dev_full' if i%5==0 else 'dev_probe') for i in range(1,21)]
        self.state['history']=hist
        torch.save(dict(self.state,model={'sample':torch.ones(2)},optimizer={}),self.root/'pilot/latest.pt')
        (self.root/'pilot/history.json').write_text(json.dumps(hist+[row(21,.9)]))
        for stage in ('preflight','overfit'):
            (self.root/stage).mkdir()
            (self.root/stage/'summary.json').write_text(json.dumps({'passed':True,'data_hash':'fixture'}))
        actual=auto.read_checkpoint(self.root)
        self.assertEqual(actual['position'],20)
        self.assertEqual(len(actual['history']),20)
        (self.root/'overfit/summary.json').write_text(json.dumps({'passed':False,'data_hash':'fixture'}))
        with self.assertRaises(ValueError):
            auto.read_checkpoint(self.root)

    def test_second_controller_refuses_lock_without_touching_existing_status(self):
        import fcntl
        (self.root/'pilot').mkdir()
        (self.root/'pilot/latest.pt').touch()
        with (self.root/'.autopilot.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=subprocess.run([sys.executable,str(PART2/'scratch_autopilot.py'),
                                   '--run_dir',str(self.root)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('already owns',result.stdout)
        self.assertFalse((self.root/'autopilot_status.json').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
