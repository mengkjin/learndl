"""The management entry must never implicitly install schedules."""
import unittest
from unittest.mock import MagicMock, patch

from src.api.calls.app import ManageTaskSchedules
from src.api.calls.launcher import DirectCallHub, _TOP_LEVEL_LABELS
from src.api.calls.source_code import GitClearPull
from src.api.util.direct_call import ProcessReload


class SchedulingMenuTest(unittest.TestCase):
    def test_menu_starts_with_preview_and_cancel_does_nothing(self):
        with patch('src.proj.util.cli.AskFor.Options', return_value=MagicMock(valid=False)) as prompt, \
             patch.object(ManageTaskSchedules, 'spawn_in_pane') as spawn:
            DirectCallHub()._dispatch_top_level('Watchdog / Schedule Management')
        self.assertEqual(prompt.call_args.args[0][0], 'Preview Changes (default)')
        self.assertTrue(prompt.call_args.kwargs['use_checkbox'])
        spawn.assert_not_called()

    def test_explicit_selection_dispatches_only_selected_action(self):
        for label, action in [('Preview Changes (default)', 'plan'), ('Install / Update Schedules', 'apply'),
                              ('View Installation Status', 'status'), ('Roll Back Last Installation', 'rollback')]:
            with self.subTest(action=action), \
                 patch('src.proj.util.cli.AskFor.Options', return_value=MagicMock(valid=True, result=label)), \
                 patch.object(ManageTaskSchedules, 'spawn_in_pane') as spawn:
                DirectCallHub()._dispatch_top_level('Watchdog / Schedule Management')
                spawn.assert_called_once_with(action=action)

    def test_git_pull_restarts_only_after_success(self):
        hub = DirectCallHub()
        with patch.object(GitClearPull, 'run') as pull:
            with self.assertRaises(ProcessReload):
                hub._dispatch_top_level('Git Pull')
            pull.assert_called_once_with()
        with patch.object(GitClearPull, 'run', side_effect=RuntimeError('pull failed')):
            with self.assertRaisesRegex(RuntimeError, 'pull failed'):
                hub._dispatch_top_level('Git Pull')

    def test_operations_are_grouped_in_non_research_menu(self):
        labels = {label for label, _, _ in DirectCallHub._source_code_entries()}
        for label in (
            'Launch Streamlit App', 'Launch Learndl Monitor', 'Watchdog / Schedule Management',
            'Kill Running Script',
        ):
            self.assertNotIn(label, _TOP_LEVEL_LABELS)
            self.assertIn(label, labels)
        with patch.object(DirectCallHub, '_pick_direct_call', return_value=ManageTaskSchedules), \
             patch('src.proj.util.cli.AskFor.Options', return_value=MagicMock(valid=True, result='View Installation Status')), \
             patch.object(ManageTaskSchedules, 'spawn_in_pane') as spawn:
            DirectCallHub()._dispatch_top_level('Non-Research Operations')
            spawn.assert_called_once_with(action='status')

    def test_direct_call_defaults_to_read_only_preview(self):
        with patch('src.api.calls.app.subprocess.run', return_value=MagicMock(returncode=0)) as execute:
            ManageTaskSchedules().run()
        self.assertEqual(execute.call_args.args[0][-2:], ['plan', '--diff'])


if __name__ == '__main__':
    unittest.main()
