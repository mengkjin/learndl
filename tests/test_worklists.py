import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.proj import PATH
from src.res.model.workflow.worklists import load, revision_for
from src.res.model.workflow.worklist_state import WorklistState


class WorklistsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.object(PATH, 'sched_worklists', self.root))
        self.enterContext(patch('src.res.model.util.training_history.git', return_value='commit'))
        self.times = {}
        self.enterContext(patch('src.res.model.workflow.worklists.git_time', side_effect=lambda p: self.times.get(p.name)))

    def write(self, name, body, timestamp):
        (self.root / name).write_text(body)
        self.times[name] = timestamp

    def test_order_is_git_time_and_name_not_main_or_mtime(self):
        self.write('main.yaml', 'fit: [main]\nshort_test: true\n', 2)
        self.write('z.yaml', 'fit: [z]\n', 1)
        self.write('a.yaml', 'fit: [a]\n', 1)
        worklist, _, errors = load(automatic=True)
        self.assertFalse(errors)
        self.assertEqual(list(WorklistState.entries(worklist)), ['a', 'z', 'main'])
        self.assertNotIn('short_test', WorklistState.options(worklist, 'a'))
        before = revision_for(worklist, 'a')
        self.write('main.yaml', 'fit: [changed]\n', 3)
        self.assertEqual(before, revision_for(load()[0], 'a'))

    def test_duplicates_bad_yaml_and_no_history_are_isolated(self):
        self.write('a.yaml', 'fit: [duplicate, valid]\n', 1)
        self.write('b.yaml', 'fit: [duplicate]\n', 2)
        self.write('bad.yaml', 'fit: [\n', 3)
        self.write('new.yaml', 'fit: [uncommitted]\n', None)
        worklist, _, errors = load(automatic=True)
        self.assertEqual(list(WorklistState.entries(worklist)), ['valid'])
        self.assertEqual(len(errors), 3)
        with self.assertRaises(ValueError):
            revision_for(worklist, 'duplicate')

    def test_deleted_file_disappears_and_standalone_defaults(self):
        self.write('main.yaml', 'fit: [main]\nresume: true\n', 1)
        self.assertTrue(WorklistState.options(load()[0], 'standalone')['resume'])
        (self.root / 'main.yaml').unlink()
        self.assertEqual(WorklistState.entries(load()[0]), {})
        self.assertFalse(WorklistState.options(load()[0], 'standalone')['resume'])
