import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock,patch
from src.automation.scheduling import factor_sequence_restart as restart
from src.automation.scheduling import factor_sequence as queue


class RestartTests(unittest.TestCase):
    def setup_paths(self,tmp):
        root=Path(tmp);models=root/'models';models.mkdir();results=root/'results';results.mkdir()
        run=restart.LEGACY_RUNS[1]
        name=f'fs_{run}_f000_20170103_factorseq_mlp_full_17'
        evidence=results/f'{run}_f000_20170103'/'training';evidence.mkdir(parents=True)
        (evidence/'runs.json').write_text(json.dumps([dict(name=name,model='factorseq_mlp')]))
        target=models/f'factorseq_mlp@{name}';target.mkdir();(target/'weights').write_text('old')
        return root,models,results,target

    def test_only_recorded_old_fold_roots_deleted_without_backup(self):
        with TemporaryDirectory() as tmp:
            root,models,results,target=self.setup_paths(tmp)
            unrelated=models/'factorseq_mlp@fs_new_schedule';unrelated.mkdir()
            targets=restart.plan_cleanup(models,results)
            self.assertEqual(len(targets),1)
            report=restart.delete_planned(targets,root/'cleanup.json')
            self.assertFalse(target.exists());self.assertTrue(unrelated.exists())
            self.assertEqual(report['status'],'passed');self.assertFalse(report['backup'])
            self.assertEqual(restart.plan_cleanup(models,results),[])

    def test_missing_evidence_and_links_refuse_before_deletion(self):
        with TemporaryDirectory() as tmp:
            root,models,results,target=self.setup_paths(tmp)
            (target/'linked').symlink_to(root/'elsewhere')
            with self.assertRaises(ValueError):
                restart.plan_cleanup(models,results)
            self.assertTrue(target.exists());(target/'linked').unlink()
            next(results.glob('*/training/runs.json')).unlink()
            with self.assertRaises(ValueError):
                restart.plan_cleanup(models,results)
            self.assertTrue(target.exists())

    def test_changed_identity_rejected(self):
        with TemporaryDirectory() as tmp:
            root,models,results,target=self.setup_paths(tmp)
            targets=restart.plan_cleanup(models,results)
            targets[0]['inode']+=1
            with self.assertRaises(ValueError):
                restart.delete_planned(targets,root/'cleanup.json')
            self.assertTrue(target.exists())
            self.assertEqual(json.loads((root/'cleanup.json').read_text())['status'],'failed')

    def test_request_is_durable_and_not_retried(self):
        with TemporaryDirectory() as tmp,patch.object(queue,'requests_dir',return_value=Path(tmp)):
            path=queue.submit('new',cleanup_legacy=True)
            self.assertEqual(json.loads(path.read_text())['schema_version'],3)
            self.assertEqual(queue.submit('new',cleanup_legacy=True),path)
            store=Mock();store.owned.return_value=[]
            request=queue.observe(store)
            self.assertTrue(request['revisions']['cleanup_legacy'])
            store.owned.return_value=[{**request,'phase':'killed'}]
            self.assertIsNone(queue.observe(store))

    def test_active_original_prevents_cleanup_and_training(self):
        from contextlib import nullcontext
        with TemporaryDirectory() as tmp:
            root,models,results,target=self.setup_paths(tmp)
            result_root=root/'out';result_root.mkdir()
            (result_root/'factor_sequence').symlink_to(results,target_is_directory=True)
            module=Mock()
            with patch('src.proj.PATH.result',result_root),patch('src.proj.PATH.model_nn',models), \
                 patch('src.proj.util.script.FitLockNN.guard',return_value=nullcontext()), \
                 patch('src.automation.scheduling.factor_sequence_progress.collect',return_value={'scheduler':[]}), \
                 patch('src.automation.scheduling.factor_sequence_recovery.original_inactive',return_value=False), \
                 patch('src.proj.util.web.task_mail.TaskResultMail'):
                with self.assertRaisesRegex(RuntimeError,'still alive'):
                    restart.execute('new',module)
            self.assertTrue(target.exists());module.main.assert_not_called()

    def test_surviving_managed_child_blocks_cleanup(self):
        import sqlite3
        with TemporaryDirectory() as tmp:
            database=Path(tmp)/'runs.sqlite'
            with sqlite3.connect(database) as conn:
                conn.execute('CREATE TABLE runs(data TEXT)')
                conn.execute('INSERT INTO runs VALUES (?)',(json.dumps(dict(request_id='old',task_kind='factor_sequence',pid=123)),))
            with patch('src.automation.scheduling.runtime.members',return_value=[Mock()]):
                with self.assertRaisesRegex(RuntimeError,'still has members'):
                    restart.check_managed_scopes('old',tmp)
            with patch('src.automation.scheduling.runtime.members',return_value=[]):
                restart.check_managed_scopes('old',tmp)

    def test_cleanup_precedes_one_new_six_model_suite(self):
        from contextlib import nullcontext
        with TemporaryDirectory() as tmp:
            root,models,results,target=self.setup_paths(tmp)
            result_root=root/'out';result_root.mkdir()
            (result_root/'factor_sequence').symlink_to(results,target_is_directory=True)
            module=Mock()
            def start(**kwargs):
                self.assertFalse(target.exists())
                self.assertEqual(kwargs,dict(run_id='new',test_start=20170101,research=False,ablations=False))
                return 0
            module.main.side_effect=start
            with patch('src.proj.PATH.result',result_root),patch('src.proj.PATH.model_nn',models), \
                 patch('src.proj.util.script.FitLockNN.guard',return_value=nullcontext()), \
                 patch('src.automation.scheduling.factor_sequence_progress.collect',return_value={'scheduler':[]}), \
                 patch('src.automation.scheduling.factor_sequence_recovery.original_inactive',return_value=True), \
                 patch.object(restart,'check_managed_scopes'), \
                 patch('src.proj.util.web.task_mail.TaskResultMail') as mail:
                mail.return_value.return_value=True
                self.assertEqual(restart.execute('new',module),0)
            module.main.assert_called_once()
