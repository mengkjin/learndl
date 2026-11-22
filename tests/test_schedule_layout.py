"""Regression coverage for schedule aliases and the one-time layout move."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.proj import PATH
from src.proj.env.schedules import available_schedules, find_schedule_path
from src.res.model.util.schedule_output import save_model_source, validate_model_source
from src.res.model.workflow.schedule_revision import schedule_revision
from src.res.model.workflow.worklist_state import WorklistState
from src.res.model.workflow.worklists import load
from src.res.model.util.training_history import file_revision


class ScheduleLayoutTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for key, relative in {
            'main': '.', 'sched': 'configs/schedule/current', 'sched_linear': 'configs/schedule/linear',
            'sched_shared': 'shared', 'sched_worklists': 'worklists', 'lc_machine': 'machine',
        }.items():
            path = self.root / relative
            path.mkdir(parents=True, exist_ok=True)
            self.enterContext(patch.object(PATH, key, path))

    @staticmethod
    def write(path, content='{}\n'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def commit(self, message):
        self.git('add', '-A')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def test_recursive_lookup_and_ambiguity(self):
        path = self.write(PATH.sched / 'a/b/model.yaml')
        self.write(PATH.sched_linear / 'top.yaml')
        self.write(PATH.sched.parent / 'archive/model.yaml')
        self.assertEqual(find_schedule_path('model'), path)
        self.assertEqual(find_schedule_path('a/b/model'), path)
        self.assertEqual(available_schedules(), ['a/b/model', 'top'])
        self.write(PATH.sched_linear / 'other/model.yaml')
        with self.assertRaisesRegex(ValueError, 'Duplicate schedule'):
            find_schedule_path('model')
        self.assertEqual(find_schedule_path('a/b/model'), path)
        self.write(PATH.sched_shared / 'a/b/model.yaml')
        with self.assertRaisesRegex(ValueError, 'Duplicate schedule'):
            find_schedule_path('a/b/model')

    def test_path_escape_is_rejected(self):
        outside = self.write(self.root / 'outside.yaml')
        (PATH.sched / 'escape.yaml').symlink_to(outside)
        for name in ('escape', '../outside', '/outside', 'a//b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                find_schedule_path(name)
        self.assertNotIn('escape', available_schedules())

    def test_aliases_share_worklist_identity_and_cannot_run_twice(self):
        self.write(PATH.sched / 'nested/model.yaml')
        self.write(PATH.sched_worklists / 'main.yaml', 'fit: [model]\n')
        self.write(PATH.sched_worklists / 'other.yaml', 'fit: [nested/model]\n')
        self.assertEqual(WorklistState('nested/model').path, WorklistState('model').path)
        with patch('src.res.model.workflow.worklists.git_time', return_value=1):
            worklist, _, errors = load(automatic=True)
        self.assertEqual(WorklistState.entries(worklist), {})
        self.assertIn('nested/model', errors[0])

    def test_standalone_request_id_survives_alias_change(self):
        self.write(PATH.sched / 'nested/model.yaml')
        first = WorklistState('model').request_training('experiment', allow_unlisted=True)
        alias = WorklistState('nested/model').request_training('experiment', allow_unlisted=True)
        self.assertEqual(first, alias)
        self.assertEqual(WorklistState.requested_schedules(), ['model'])

    def test_pure_move_preserves_revision_and_real_edits_invalidate_it(self):
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        old = self.write(PATH.sched / 'model.yaml')
        base = self.commit('old')
        before = file_revision(old)
        new = PATH.sched / 'nested/model.yaml'
        new.parent.mkdir()
        old.rename(new)
        manifest = self.root / 'configs/schedule/layout_migration.json'
        manifest.write_text(json.dumps({'base': base, 'moves': {new.relative_to(self.root).as_posix(): {
            'old': old.relative_to(self.root).as_posix(),
            'sha256': hashlib.sha256(new.read_bytes()).hexdigest(),
        }}}))
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            schedule_revision(new)
        self.commit('move')
        self.assertEqual(schedule_revision(new), before)
        self.write(new, '# edit\n{}\n')
        self.commit('edit')
        self.assertNotEqual(schedule_revision(new), before)
        self.write(new)
        self.commit('revert')
        self.assertNotEqual(schedule_revision(new), before)

    def test_same_stem_cannot_claim_another_schedules_output(self):
        first = self.write(PATH.sched / 'a/model.yaml')
        second = self.write(PATH.sched / 'b/model.yaml')
        output = self.root / 'output'
        output.mkdir()
        model = SimpleNamespace(model_clean_name='model', base=output, conf=lambda: output)
        with self.assertRaisesRegex(ValueError, 'legacy model'):
            validate_model_source(model, first)
        save_model_source(model, first)
        validate_model_source(model, first)
        with self.assertRaisesRegex(ValueError, 'belongs to current/a/model'):
            validate_model_source(model, second)
