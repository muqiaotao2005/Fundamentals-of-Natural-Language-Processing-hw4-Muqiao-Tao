"""Offline export checks with actual tokenization/SQL execution and a fixed decoder."""
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PART2 = ROOT / 'part2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(PART2))
import torch
from load_data import get_tokenizer, INPUT_PREFIX
import export_scratch_test as exporter


class FixedDecoder:
    def __init__(self, tokenizer, mapping):
        self.tokenizer, self.mapping = tokenizer, mapping
        self.seen = []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, decoder_input_ids, **kwargs):
        assert kwargs == dict(do_sample=False, num_beams=1, max_new_tokens=512)
        assert decoder_input_ids.shape == (len(input_ids), 1)
        assert not decoder_input_ids.any()
        questions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        self.seen.extend(questions)
        encoded = self.tokenizer([self.mapping[q] for q in questions], padding=True,
                                 return_tensors='pt')['input_ids']
        return torch.cat([torch.zeros((len(encoded),1),dtype=torch.long),encoded],dim=1)


class ExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = get_tokenizer()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.data = self.root / 'data'
        self.data.mkdir()
        (self.root/'records').mkdir()
        (self.run/'pilot/best').mkdir(parents=True)
        (self.run/'pilot/predictions').mkdir()
        self.config = dict(stage='pilot', initialization='all learned weights random; config only',
                           data_hash='fixture', seed=42, eval_batch_size=2,
                           max_new_tokens=512, sql_timeout=.2, sql_workers=2)
        self.best = dict(epoch=35, record_f1=.61, evaluation_scope='dev_full')
        self.history = [dict(epoch=30,record_f1=.60,evaluation_scope='dev_full'),self.best,
                        dict(epoch=36,record_f1=.99,evaluation_scope='dev_probe'),
                        dict(epoch=45,record_f1=.58,evaluation_scope='dev_full')]
        self.stop = dict(status='stopped',best_epoch=35,best_full_dev_record_f1=.61,completed_epochs=45)
        for path, value in [('pilot/run_config.json',self.config),('pilot/history.json',self.history),
                            ('pilot/best_metrics.json',self.best),('autopilot_summary.json',self.stop),
                            ('autopilot_status.json',self.stop)]:
            (self.run/path).write_text(json.dumps(value))
        (self.run/'pilot/best/model.safetensors').write_bytes(b'fixture: local checkpoint')
        (self.data/'dev.nl').write_text('give two\ngive one\n')
        (self.data/'dev.sql').write_text('SELECT 2\nSELECT 1\n')
        (self.data/'test.nl').write_text('give three\ngive empty\ngive three again\n')
        # No test SQL or hidden reference file exists.
        (self.run/'pilot/predictions/epoch_035_dev_full.sql').write_text('SELECT 2\nSELECT 1\n')
        with (self.root/'records/ground_truth_dev.pkl').open('wb') as out:
            pickle.dump(([[(2,)],[(1,)]],['','']),out)
        conn=sqlite3.connect(self.data/'flight_database.db')
        conn.execute('CREATE TABLE preserved (x INTEGER)')
        conn.commit()
        conn.close()
        self.db_hash=exporter.sha256(self.data/'flight_database.db')
        self.model=FixedDecoder(self.tokenizer,{
            INPUT_PREFIX+'give two':'SELECT 2', INPUT_PREFIX+'give one':'SELECT 1',
            INPUT_PREFIX+'give three':'SELECT 3', INPUT_PREFIX+'give empty':'',
            INPUT_PREFIX+'give three again':'SELECT 3'})
        self.args=SimpleNamespace(run_dir=self.run,data_dir=self.data,device=torch.device('cpu'))

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self):
        with patch.object(exporter,'DEV_EXAMPLES',2),patch.object(exporter,'TEST_EXAMPLES',3),\
             patch.object(exporter,'fingerprint',return_value='fixture'),\
             patch.object(exporter,'load_saved_model',return_value=(self.model,self.tokenizer)) as loaded:
            result=exporter.export(self.args)
            loaded.assert_called_once_with(self.run/'pilot/best',torch.device('cpu'))
            return result

    def test_selects_best_full_dev_not_last_or_probe(self):
        _,best=exporter.selected_run(self.run)
        self.assertEqual(best['epoch'],35)
        (self.run/'pilot/best_metrics.json').write_text(json.dumps(self.history[-1]))
        with self.assertRaises(ValueError):
            exporter.selected_run(self.run)

    def test_refuses_export_before_training_stops(self):
        (self.run/'autopilot_status.json').write_text(json.dumps({'status':'training'}))
        with self.assertRaises(ValueError):
            exporter.selected_run(self.run)

    def test_export_keeps_order_empty_prediction_pickle_shape_and_only_ec_zip_files(self):
        baseline=self.root/'t5_ft_experiment_test.sql'
        baseline.write_text('baseline must remain\n')
        result=self.invoke()
        folder=self.run/'final'
        self.assertTrue(result['dev_sql_reproduced_exactly'])
        self.assertIsNone(result['test_record_f1'])
        self.assertFalse(result['test_labels_used'])
        self.assertEqual(result['test_execution_error_count'],1)
        self.assertEqual((folder/(exporter.SUBMISSION_STEM+'.sql')).read_text(),'SELECT 3\n\nSELECT 3\n')
        with (folder/(exporter.SUBMISSION_STEM+'.pkl')).open('rb') as source:
            payload=pickle.load(source)
        self.assertIsInstance(payload,tuple)
        self.assertEqual(payload[0],[[(3,)],[],[(3,)]])
        self.assertEqual([bool(x) for x in payload[1]],[False,True,False])
        self.assertEqual(self.model.seen,[INPUT_PREFIX+x for x in
                         ['give two','give one','give three','give empty','give three again']])
        with zipfile.ZipFile(folder/'submission.zip') as archive:
            self.assertEqual(archive.namelist(),[exporter.SUBMISSION_STEM+'.sql',exporter.SUBMISSION_STEM+'.pkl'])
            self.assertIsNone(archive.testzip())
        self.assertEqual(self.db_hash,exporter.sha256(self.data/'flight_database.db'))
        self.assertEqual(baseline.read_text(),'baseline must remain\n')
        with self.assertRaises(ValueError):
            self.invoke()

    def test_dev_reproduction_failure_prevents_any_test_access(self):
        (self.run/'pilot/predictions/epoch_035_dev_full.sql').write_text('SELECT 9\nSELECT 1\n')
        (self.data/'test.nl').unlink()
        with self.assertRaisesRegex(ValueError,'test was not generated'):
            self.invoke()
        self.assertFalse((self.run/'final').exists())
        self.assertEqual(self.model.seen,[INPUT_PREFIX+'give two',INPUT_PREFIX+'give one'])

    def test_changed_fingerprint_stops_before_generation(self):
        with patch.object(exporter,'fingerprint',return_value='changed'),\
             patch.object(exporter,'load_saved_model',return_value=(self.model,self.tokenizer)):
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):
                exporter.export(self.args)
        self.assertEqual(self.model.seen,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
