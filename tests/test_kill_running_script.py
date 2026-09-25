"""Kill Running Script lists live tasks and kills only after confirmation."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.calls.script import KillRunningScript
from src.proj.util.cli.ask import AskFlag


def _task(task_id: str, pid: int, script: str, start: str = '2026-09-24 23:01:03') -> MagicMock:
    item = MagicMock()
    item.id = task_id
    item.pid = pid
    item.script = script
    item.script_key = script
    item.is_running = True
    item.time_str.return_value = start
    item.kill.return_value = True
    return item


def _label(item: MagicMock, start: str = '2026-09-24 23:01:03') -> str:
    return f'PID {item.pid} | {Path(item.script).name} | {start}'


class _Query:
    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = batches
        self.index = 0

    def __enter__(self):
        task_ids = self.batches[self.index]
        self.index += 1
        cursor = MagicMock()
        cursor.fetchall.return_value = [{'task_id': task_id} for task_id in task_ids]
        return MagicMock(), cursor

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class KillRunningScriptTest(unittest.TestCase):
    def _run(self, batches, items, options, confirmation=None, status=None):
        db = MagicMock()
        db.conn_handler = _Query(batches)
        by_id = {item.id: item for item in items}

        def load(task_id, _db):
            return by_id[task_id]

        confirmation = confirmation or MagicMock(side_effect=AssertionError('confirmation not expected'))
        status = status or (lambda _pid: 'running')
        with patch('src.api.util.backend.task.TaskDatabase', return_value=db), \
             patch('src.api.util.backend.task.TaskItem.load', side_effect=load), \
             patch('src.proj.util.shell.util.process.check_status', side_effect=status), \
             patch('src.proj.util.cli.ask.AskFor.Options', side_effect=options), \
             patch('src.proj.util.cli.ask.AskFor.Confirmation', confirmation):
            KillRunningScript().run()
        return db

    def test_refresh_reloads_the_menu_and_hides_dead_pids(self) -> None:
        own = os.getpid()
        live = _task('4_train/2_schedule_model.py@1', 111 if own != 111 else 112, '/repo/scripts/4_train/2_schedule_model.py')
        dead = _task('4_train/1_other.py@2', 222 if own != 222 else 223, '/repo/scripts/4_train/1_other.py')
        mine = _task('4_train/3_self.py@3', own, '/repo/scripts/4_train/3_self.py')
        later = _task('1_data/0_update.py@4', 333 if own != 333 else 334, '/repo/scripts/1_data/0_update.py', '2026-09-24 23:05:00')
        menus: list[list[str]] = []

        def options(labels, **_kwargs):
            menus.append(list(labels))
            if len(menus) == 1:
                return AskFlag('valid').set_result([KillRunningScript._REFRESH_LABEL])
            return AskFlag('exit')

        def status(pid: int) -> str:
            return 'complete' if pid == dead.pid else 'running'

        confirm = MagicMock(side_effect=AssertionError('confirmation not expected'))
        self._run(
            [[live.id, dead.id, mine.id], [later.id]],
            [live, dead, mine, later],
            options,
            confirmation=confirm,
            status=status,
        )
        self.assertEqual(menus[0][0], 'Refresh')
        self.assertIn(_label(live), menus[0])
        self.assertNotIn(_label(dead), menus[0])
        self.assertNotIn(_label(mine), menus[0])
        self.assertEqual(menus[1], ['Refresh', _label(later, '2026-09-24 23:05:00')])
        confirm.assert_not_called()
        live.kill.assert_not_called()

    def test_rejected_confirmation_does_not_kill(self) -> None:
        item = _task('4_train/2_schedule_model.py@1', 424242, '/repo/scripts/4_train/2_schedule_model.py')
        label = _label(item)
        prompts = {'n': 0}

        def options(labels, **_kwargs):
            prompts['n'] += 1
            if prompts['n'] == 1:
                self.assertEqual(labels[0], 'Refresh')
                return AskFlag('valid').set_result([label])
            return AskFlag('exit')

        confirm = MagicMock(return_value=AskFlag('exit'))
        self._run([[item.id], []], [item], options, confirmation=confirm)
        confirm.assert_called_once()
        item.kill.assert_not_called()

    def test_confirmed_kill_requires_matching_live_pid(self) -> None:
        item = _task('4_train/2_schedule_model.py@1', 424242, '/repo/scripts/4_train/2_schedule_model.py')
        label = _label(item)

        def options(_labels, **_kwargs):
            if options.calls == 0:
                options.calls += 1
                return AskFlag('valid').set_result([label])
            return AskFlag('exit')

        options.calls = 0
        confirm = MagicMock(return_value=AskFlag('valid'))
        self._run([[item.id], []], [item], options, confirmation=confirm)
        item.kill.assert_called_once_with()
        updates = item.update.call_args.args[0]
        self.assertEqual(updates['status'], 'killed')
        self.assertEqual(updates['exit_error'], 'Killed by operator from CLI')
        self.assertTrue(item.update.call_args.kwargs['sync'])

    def test_confirmed_kill_skips_when_pid_changes(self) -> None:
        listed = _task('4_train/2_schedule_model.py@1', 424242, '/repo/scripts/4_train/2_schedule_model.py')
        changed = _task(listed.id, 525252, listed.script)
        label = _label(listed)
        loads = iter((listed, changed))

        def options(_labels, **_kwargs):
            if options.calls == 0:
                options.calls += 1
                return AskFlag('valid').set_result([label])
            return AskFlag('exit')

        options.calls = 0

        def load(_task_id, _db):
            return next(loads)

        db = MagicMock()
        db.conn_handler = _Query([[listed.id], []])
        with patch('src.api.util.backend.task.TaskDatabase', return_value=db), \
             patch('src.api.util.backend.task.TaskItem.load', side_effect=load), \
             patch('src.proj.util.shell.util.process.check_status', return_value='running'), \
             patch('src.proj.util.cli.ask.AskFor.Options', side_effect=options), \
             patch('src.proj.util.cli.ask.AskFor.Confirmation', return_value=AskFlag('valid')):
            KillRunningScript().run()
        listed.kill.assert_not_called()
        changed.kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
