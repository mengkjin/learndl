"""The management entry must never implicitly install schedules."""
import unittest
from unittest.mock import MagicMock, patch

from src.api.calls.app import ManageTaskSchedules
from src.api.calls.launcher import DirectCallHub


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

    def test_direct_call_defaults_to_read_only_preview(self):
        with patch('src.api.calls.app.subprocess.run', return_value=MagicMock(returncode=0)) as execute:
            ManageTaskSchedules().run()
        self.assertEqual(execute.call_args.args[0][-2:], ['plan', '--diff'])


if __name__ == '__main__':
    unittest.main()
