"""Training provenance and worklist policy tests without datasets or GPUs."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil
import torch

from src.api.calls.research import CarryOutScheduleWorkList
from src.api.calls.worklist_state import WorklistState
from src.proj import PATH
from src.res.model.util import training_history as history
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import ModelPath
from src.res.model.util.resume_validation import validate_resume


def fake_config(folder):
    return SimpleNamespace(
        base_path=SimpleNamespace(base=folder), model_name='example',
        model_config=SimpleNamespace(schedule_name='example', Param={'seed': 42}),
        schedule_config=SimpleNamespace(Param={'model.module': 'gru'}),
        algo_config=SimpleNamespace(Param={'hidden': 8}), boost_head_config=None,
        queue_of_stages=['data', 'fit', 'test'], is_resuming=False, short_test=False,
    )


class FakeTrainer:
    def __init__(self, folder):
        self._config_kwargs = {'module': 'gru'}
        self.input_model_kwargs = {'resume': 0}
        self.config = fake_config(folder)

    @history.record_training
    def execute(self, error=None):
        self.training_run.configured(self.config)
        if error:
            raise error
        return self


class HistoryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(PATH, 'lc_machine', self.root))
        self.enterContext(patch.object(history, 'git', return_value='test-git'))
        self.wait = self.enterContext(patch.object(history.Save, 'async_wait_all'))

    def test_success_and_effective_config(self):
        with history.collect_training_runs() as runs:
            trainer = FakeTrainer(self.root / 'model').execute()
        self.assertEqual(runs[0]['status'], 'success')
        self.assertEqual(runs[0]['model_config'], {'seed': 42})
        self.assertEqual(runs[0]['git']['head'], 'test-git')
        self.assertIsNone(trainer.training_run)
        self.wait.assert_called_once()
        saved = json.loads(next((self.root / 'training_history').glob('*/run.json')).read_text())
        self.assertEqual(saved['run_id'], runs[0]['run_id'])

    def test_errors_interruptions_and_async_save_failure(self):
        for error, status in [(ValueError('failed'), 'failed'), (KeyboardInterrupt(), 'interrupted')]:
            with self.subTest(status=status), history.collect_training_runs() as runs:
                with self.assertRaises(type(error)):
                    FakeTrainer(self.root).execute(error)
                self.assertEqual(runs[0]['status'], status)
        self.wait.side_effect = OSError('disk failed')
        with history.collect_training_runs() as runs:
            with self.assertRaises(OSError):
                FakeTrainer(self.root).execute()
        self.assertEqual(runs[0]['status'], 'failed')

    def test_dead_process_is_interrupted(self):
        run = history.TrainingRun(FakeTrainer(self.root))
        with patch.object(history.psutil, 'Process', side_effect=psutil.NoSuchProcess(123)):
            self.assertEqual(history.TrainingRun.read(run.path)['status'], 'interrupted')


class WorklistTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.worklist = self.root / 'worklist.yaml'
        self.worklist.write_text('fit: [example]\nresume: false\nforce: false\n')
        self.schedule = self.root / 'example.yaml'
        self.schedule.write_text('model.module: gru\n')
        self.folder = self.root / 'model'
        self.folder.mkdir()
        self.enterContext(patch.object(PATH, 'lc_machine', self.root))
        self.enterContext(patch.object(PATH, 'sched_worklist', self.worklist))
        self.enterContext(patch.object(history, 'git', return_value=None))
        self.enterContext(patch('src.res.model.util.config.config.ScheduleConfig.find_path', return_value=self.schedule))
        self.enterContext(patch('src.res.model.util.config.config.ScheduleConfig.check_name_exist', return_value=True))
        self.enterContext(patch('src.api.calls.research.MACHINE', SimpleNamespace(platform_server=True)))
        self.enterContext(patch('src.api.calls.research.as_script_main', side_effect=lambda *_: nullcontext()))
        self.enterContext(patch.object(history.Save, 'async_wait_all'))
        self.main = Mock(side_effect=self.train)
        self.enterContext(patch.object(CarryOutScheduleWorkList, '_load_schedule_main', return_value=self.main))
        self.enterContext(patch.object(CarryOutScheduleWorkList, '_train_one_schedule', side_effect=lambda main, **kw: main(**kw)))
        self.state = WorklistState('example')

    def train(self, **kwargs):
        FakeTrainer(self.folder).execute()
        return SimpleNamespace(success=True, execution_success=True)

    def run_hub(self):
        CarryOutScheduleWorkList().run()

    def test_success_skips_even_when_resume_true(self):
        self.worklist.write_text('fit: [example]\nresume: true\n')
        self.run_hub()
        self.run_hub()
        self.main.assert_called_once()
        self.assertEqual(self.state.read()['status'], 'success')

    def test_comment_and_schedule_edits_rerun(self):
        self.run_hub()
        self.worklist.write_text(self.worklist.read_text() + '# changed\n')
        self.run_hub()
        self.schedule.write_text('model.module: gru\n# new revision\n')
        self.run_hub()
        self.assertEqual(self.main.call_count, 3)

    def test_force_always_runs_and_preserves_resume(self):
        self.worklist.write_text('fit: [example]\nresume: true\nforce: true\n')
        self.run_hub()
        self.run_hub()
        self.assertEqual(self.main.call_count, 2)
        self.assertTrue(self.main.call_args.kwargs['resume'])
        self.assertEqual(self.main.call_args.kwargs['base_path'], str(self.folder.resolve()))

    def test_automatic_runs_one_schedule_and_force_only_once(self):
        from src.api.calls.worklist_state import AutomaticDeferred
        from src.api.task_monitor.scheduling.idle_worklist import version
        self.worklist.write_text('fit: [example, second]\nresume: false\nforce: true\n')
        expected = version(self.state.revisions('example'))
        automatic = CarryOutScheduleWorkList(automatic=True, only_schedule='example', expected_version=expected)
        automatic.run()
        self.assertEqual(self.main.call_args.kwargs['schedule_name'], 'example')
        self.assertEqual(self.main.call_args.kwargs['resume_selection'], 'latest')
        self.assertTrue(self.main.call_args.kwargs['email'])
        with self.assertRaises(AutomaticDeferred):
            automatic.run()
        self.main.assert_called_once()
        self.schedule.write_text('# changed before launch\n')
        with self.assertRaises(AutomaticDeferred):
            automatic.run()
        self.main.assert_called_once()

    def test_missing_directory_starts_new_training(self):
        self.worklist.write_text('fit: [example]\nresume: true\n')
        self.run_hub()
        self.folder.rmdir()
        self.run_hub()
        self.assertFalse(self.main.call_args.kwargs['resume'])
        self.assertNotIn('base_path', self.main.call_args.kwargs)

    def test_failed_or_swallowed_failure_is_retried(self):
        self.main.side_effect = lambda **kw: SimpleNamespace(success=False, execution_success=False)
        with self.assertRaises(RuntimeError):
            self.run_hub()
        self.assertEqual(self.state.read()['status'], 'failed')
        self.main.side_effect = self.train
        self.run_hub()
        self.assertEqual(self.state.read()['status'], 'success')

    def test_active_lock_prevents_duplicate_even_when_forced(self):
        self.worklist.write_text('fit: [example]\nresume: false\nforce: true\n')
        with self.state.lock():
            self.run_hub()
        self.main.assert_not_called()

    def test_edit_during_training_does_not_cache_completion(self):
        def edit(**kwargs):
            result = self.train(**kwargs)
            self.schedule.write_text('# modified during run\n')
            return result
        self.main.side_effect = edit
        with self.assertRaises(RuntimeError):
            self.run_hub()
        self.assertEqual(self.state.read()['status'], 'failed')

    def test_worklist_change_reruns_all_but_schedule_change_only_one(self):
        second = self.root / 'second.yaml'
        second.write_text('model.module: gru\n')
        self.worklist.write_text('fit: [example, second]\nresume: false\n')
        with patch('src.res.model.util.config.config.ScheduleConfig.find_path',
                   side_effect=lambda name: self.schedule if name == 'example' else second):
            self.run_hub()
            self.main.reset_mock()
            self.schedule.write_text('# new config\n')
            self.run_hub()
            self.main.assert_called_once()
            self.assertEqual(self.main.call_args.kwargs['schedule_name'], 'example')
            self.main.reset_mock()
            self.worklist.write_text('fit: [second, example]\nresume: false\n')
            self.run_hub()
            self.assertEqual(self.main.call_count, 2)

    def test_directory_and_run_id_are_persisted_before_completion(self):
        def train(**kwargs):
            class InspectTrainer(FakeTrainer):
                @history.record_training
                def execute(inner):
                    inner.training_run.configured(inner.config)
                    state = self.state.read()
                    self.assertEqual(state['status'], 'running')
                    self.assertEqual(state['model_path'], str(self.folder.resolve()))
                    self.assertEqual(state['training_run_ids'], [inner.training_run.run_id])
                    raise KeyboardInterrupt()
            InspectTrainer(self.folder).execute()
        self.main.side_effect = train
        with self.assertRaises(KeyboardInterrupt):
            self.run_hub()
        self.assertEqual(self.state.read()['model_path'], str(self.folder.resolve()))
        self.assertNotEqual(self.state.read()['status'], 'success')

    def test_policy_has_no_age_limit_and_respects_incomplete_state(self):
        revisions = self.state.revisions('example')
        previous = dict(status='success', model_path=str(self.folder), revisions=revisions,
                        updated_at='2000-01-01')
        self.assertEqual(self.state.decision(previous, revisions, force=False, resume=True)[0], 'completed')
        for status in ['failed', 'running', 'interrupted']:
            self.assertNotEqual(self.state.decision(previous | {'status': status}, revisions,
                                                    force=False, resume=True)[0], 'completed')
        changed = copy.deepcopy(revisions)
        changed['schedule']['git_commit'] = 'another-commit'
        self.assertNotEqual(self.state.decision(previous, changed, force=False, resume=True)[0], 'completed')


class ResumeValidationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'model.yaml').write_text('model.name: example')
        self.config = SimpleNamespace(
            is_resuming=True, queue_of_stages=['fit'], module_type='nn', submodels=['best'],
            boost_head_config=None, base_path=SimpleNamespace(
                base=self.root, is_null_model=False, conf_file=lambda key: self.root / f'{key}.yaml',
                archive=lambda: self.root / 'archive',
            ),
        )

    def test_missing_corrupt_and_valid_model(self):
        with self.assertRaisesRegex(FileNotFoundError, 'no saved models'):
            validate_resume(self.config)
        model = self.root / 'archive/0/20240101/best/state_dict.pt'
        model.parent.mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, 'saved model is missing'):
            validate_resume(self.config)
        model.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            validate_resume(self.config)
        torch.save({'weight': torch.ones(1)}, model)
        validate_resume(self.config)

    def test_new_training_does_not_require_checkpoints(self):
        self.config.is_resuming = False
        validate_resume(self.config)

    def test_new_training_uses_unoccupied_folder(self):
        base = Mock()
        base.is_short_test = False
        base.is_null_model = False
        base.base.exists.return_value = True
        base.find_resumable_candidates_indices.return_value = [1]
        base.find_new_index.return_value = 2
        config = object.__new__(ModelConfig)
        config.model_config = SimpleNamespace(base_path=base)
        config.queue_of_stages = ['fit']
        config.is_resuming = False
        config.logger = Mock()
        base.with_new_index.side_effect = lambda _: base.base.exists.configure_mock(return_value=False)
        ModelConfig.parser_select(config, 0)
        base.find_new_index.assert_called_once_with(folder_not_exist=True)
        base.with_new_index.assert_called_once_with(2)

    def test_real_schedule_config_keeps_old_directory_and_resumes_exact_path(self):
        model_root = (self.root / 'models').resolve()
        nn_root = model_root / 'nn'
        nn_root.mkdir(parents=True)
        with patch.object(PATH, 'model', model_root), patch.object(PATH, 'model_nn', nn_root), \
             patch.object(ModelPath, 'log_operation'):
            config = ModelConfig(schedule_name='gru_day_new_rtn', stage=1, resume=0,
                                 selection=0, short_test=False, vb_level='never').start_model()
            original = config.base_path.base
            # An interrupted run with configs but no checkpoints must keep its folder.
            fresh = ModelConfig(schedule_name='gru_day_new_rtn', stage=1, resume=0,
                                selection=0, short_test=False, vb_level='never').start_model()
            self.assertNotEqual(fresh.base_path.base, original)
            self.assertTrue(original.is_dir())
            with self.assertRaisesRegex(FileNotFoundError, 'no saved models'):
                ModelConfig(original, schedule_name='gru_day_new_rtn', stage=1, resume=1,
                            selection=0, short_test=False, vb_level='never').start_model()
            artifact = original / 'archive/0/20240101/best/state_dict.pt'
            artifact.parent.mkdir(parents=True)
            torch.save({'weight': torch.ones(1)}, artifact)
            resumed = ModelConfig(original, schedule_name='gru_day_new_rtn', stage=1, resume=1,
                                  selection=0, short_test=False, vb_level='never').start_model()
            self.assertEqual(resumed.base_path.base, original)
            with history.collect_training_runs() as runs, patch.object(PATH, 'lc_machine', self.root), \
                 patch.object(history, 'git', return_value=None), patch.object(history.Save, 'async_wait_all'):
                trainer = FakeTrainer(original)
                trainer.config = resumed
                trainer.execute()
            self.assertEqual(runs[0]['schedule_name'], 'gru_day_new_rtn')
            self.assertTrue(runs[0]['resume'])


class GitRevisionTest(unittest.TestCase):
    def test_unrelated_commit_does_not_invalidate_and_file_change_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', tmp, *args], check=True, capture_output=True)
            git('init')
            git('config', 'user.email', 'test@example.invalid')
            git('config', 'user.name', 'Test')
            source = root / 'schedule.yaml'
            source.write_text('model: gru\n')
            git('add', '.')
            git('commit', '-m', 'initial')
            with patch.object(PATH, 'main', root):
                first = history.file_revision(source)
                (root / 'other.py').write_text('# code changed')
                git('add', '.')
                git('commit', '-m', 'unrelated')
                self.assertEqual(history.file_revision(source), first)
                source.write_text(source.read_text() + '# comment\n')
                self.assertNotEqual(history.file_revision(source)['sha256'], first['sha256'])
                git('add', '.')
                git('commit', '-m', 'schedule changed')
                self.assertNotEqual(history.file_revision(source)['git_commit'], first['git_commit'])


if __name__ == '__main__':
    unittest.main()
