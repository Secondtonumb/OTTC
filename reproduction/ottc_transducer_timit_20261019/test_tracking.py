"""Exercise privacy filtering and resumable production identity without uploads."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from .common import Tracking


class TrackingTest(unittest.TestCase):
    @patch.dict(os.environ,{'OTTC_EVAL_UPLOAD':'0'})
    def test_probe_is_local_even_with_online_training_default(self):
        with tempfile.TemporaryDirectory() as tmp, patch('wandb.init') as init:
            output=Path(tmp)
            tracker=Tracking(output,dict(profile=True,tiny=False),mode='online')
            tracker.log(dict(step=1,train_loss=2.))
            tracker.finish()
            init.assert_not_called()
            self.assertFalse((output/'wandb_run.json').exists())
            self.assertEqual(json.loads((output/'history.jsonl').read_text())['train_loss'],2.)

    @patch.dict(os.environ,{'OTTC_EVAL_UPLOAD':'0'})
    def test_only_training_scalars_go_online_and_resume_keeps_identity(self):
        calls=[]
        def initialize(**options):
            run=MagicMock()
            run.id=options['id']
            run.settings=SimpleNamespace(mode=options['mode'])
            calls.append((options,run))
            return run
        with tempfile.TemporaryDirectory() as tmp, patch('wandb.init',side_effect=initialize), patch('wandb.Artifact') as artifact:
            output=Path(tmp)
            contract=dict(profile=False,tiny=False,train_limit=0,eval_limit=0)
            tracker=Tracking(output,contract,'online')
            tracker.log(dict(step=1,train_loss=2.,learning_rate=.001))
            tracker.log(dict(step=1,**{'dev/raw_PER_percent':40.}))
            tracker.log(dict(step=1,**{'benchmark/best_dev/test/raw_PER_percent':42.}))
            tracker.finish(complete=True)
            self.assertEqual(calls[0][1].log.call_count,1)
            self.assertEqual(calls[0][0]['settings'].console,'off')
            self.assertTrue(calls[0][0]['settings'].disable_code)
            self.assertEqual(len((output/'history.jsonl').read_text().splitlines()),3)
            artifact.assert_not_called()
            resumed=Tracking(output,contract,'online')
            self.assertEqual(calls[0][0]['id'],calls[1][0]['id'])
            self.assertEqual(calls[1][0]['resume'],'allow')
            resumed.finish()


if __name__=='__main__':
    unittest.main()
