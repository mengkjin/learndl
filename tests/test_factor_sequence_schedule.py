import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
import yaml
from src.data.preprocess.factor_sequence import schedule_training as schedule
from src.res.model.util.data.factor_sequence_plan import validate_folds,select_fold


class ScheduleTests(unittest.TestCase):
    def test_model_dates_and_splits_are_exact(self):
        folds=[dict(model_date=10,train=[1,2],valid=[5,6],test=[11,12]),
               dict(model_date=12,train=[3,4],valid=[8,9],test=[13,14])]
        models,dates=validate_folds(folds,np.arange(10,15))
        self.assertEqual(models.tolist(),[10,12]);self.assertEqual(dates.tolist(),[11,12,13,14])
        self.assertEqual(select_fold(folds,12)['train'],[3,4])
        with self.assertRaises(ValueError):
            validate_folds(folds,np.arange(10,16))
        with self.assertRaises(ValueError):
            select_fold(folds,13)
        config=schedule.schedule_config('/tmp/data','one','factorseq_m1',folds)
        self.assertEqual(config['model.beg_date'],10)
        self.assertEqual(config['model.end_date'],14)
        self.assertEqual(config['train.factor_sequence.folds'],folds)
        self.assertEqual(config['train.factor_sequence.train_dates'],[])

    def test_two_folds_use_one_normal_schedule_call_and_one_root(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);suite=root/'suite';suite.mkdir()
            folds=[dict(train=[1],valid=[5],test=[11,12]),dict(train=[3],valid=[7],test=[13,14])]
            (suite/'walk_forward.json').write_text(json.dumps({'folds':folds}))
            main=Mock(return_value=SimpleNamespace(success=True,error_messages=[]))
            pred=pd.DataFrame({'date':[11,12,13,14],'secid':[1]*4,'pred':[.1]*4,'label':[.2]*4})
            with patch('src.proj.PATH.sched_shared',root/'schedules'), \
                 patch('src.proj.CALENDAR.td',side_effect=lambda d,n:d+n), \
                 patch('src.res.model.util.schedule_resume.find_schedule_runs',side_effect=[[],[SimpleNamespace(base=root/'single_model')]]), \
                 patch('src.proj.util.filesys.dynamic_import.dynamic_modules',return_value=[SimpleNamespace(main=main)]), \
                 patch.object(schedule,'load',return_value={'version':'v'}), \
                 patch.object(schedule,'read_predictions',return_value=pred), \
                 patch('src.proj.Save.async_wait_all'):
                result=schedule.train(root/'data',suite,'factorseq_m1',17,{})
            main.assert_called_once()
            self.assertFalse(main.call_args.kwargs['resume'])
            self.assertTrue(main.call_args.kwargs['email'])
            self.assertEqual(result['model_path'],str(root/'single_model'))
            self.assertEqual(result['folds'],2)
            configs=list((root/'schedules').glob('*.yaml'));self.assertEqual(len(configs),1)
            self.assertEqual(len(yaml.safe_load(configs[0].read_text())['train.factor_sequence.folds']),2)
            resumed=root/'resume';resumed.mkdir()
            (resumed/'walk_forward.json').write_text(json.dumps({'folds':folds}))
            with patch('src.proj.PATH.sched_shared',root/'schedules'), \
                 patch('src.proj.CALENDAR.td',side_effect=lambda d,n:d+n), \
                 patch('src.res.model.util.schedule_resume.find_schedule_runs',return_value=[SimpleNamespace(base=root/'single_model')]), \
                 patch('src.proj.util.filesys.dynamic_import.dynamic_modules',return_value=[SimpleNamespace(main=main)]), \
                 patch.object(schedule,'load',return_value={'version':'v'}), \
                 patch.object(schedule,'read_predictions',return_value=pred), \
                 patch('src.proj.Save.async_wait_all'):
                reused=schedule.train(root/'data',resumed,'factorseq_m1',17,{},resume_suite=suite)
            main.assert_called_once() # Completed model is not trained/emailed again.
            self.assertEqual(reused['model_path'],result['model_path'])
            self.assertEqual(len(list((root/'schedules').glob('*.yaml'))),1)


    def test_real_schedule_config_retains_folds_and_replaces_scheduler(self):
        from src.res.model.util.config.config import ModelConfig
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            folds=[dict(model_date=20161230,train=[20150105],valid=[20161101],test=[20170103])]
            config=schedule.schedule_config('/tmp/data','fs_test_schedule','factorseq_m1',folds)
            (folder/'fs_test_schedule.yaml').write_text(yaml.safe_dump(config))
            with patch('src.proj.PATH.sched_shared',folder):
                loaded=ModelConfig(schedule_name='fs_test_schedule',override={'env.short_test':False})
            self.assertEqual(loaded.Param.get('train.factor_sequence.folds'),folds)
            self.assertEqual(loaded.Param.get('train.trainer.scheduler'),{'name':'step','param':{'step_size':1000,'gamma':1.}})
