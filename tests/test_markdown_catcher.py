"""Markdown archives survive shared-folder loss and preserve failed local exports."""
from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.proj.env import PATH
from src.proj.util.catcher import CrashProtectorCatcher, HtmlCatcher, MarkdownCatcher
from src.proj.util.script.autorun import AutoRunCatchers


class MarkdownCatcherTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.enterContext(patch.object(PATH, 'share_folder', self.root / 'share'))
        self.enterContext(patch.object(MarkdownCatcher, 'export_dir', self.root / 'archive'))
        self.enterContext(patch.object(HtmlCatcher, 'export_dir', self.root / 'html'))
        self.enterContext(patch.object(CrashProtectorCatcher, 'export_dir', self.root / 'runtime'))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))

    def test_shared_log_is_renamed_and_local_archive_uses_crash_log(self):
        with patch('src.proj.util.catcher.crash_protector.shutil.copyfile', wraps=shutil.copyfile) as copy:
            with CrashProtectorCatcher('task') as crash:
                with MarkdownCatcher('training', to_share_folder=True, crash_protector=crash) as md:
                    print('training result')
                    # Distinct content proves which file supplies the local archive.
                    crash.markdown_writer.write('only in local crash log')
                    running = md.running_filename
                self.assertTrue(crash.filename.exists())
                local = md.export_file_list[0].read_text()
                shared = md.share_filename.read_text()
                self.assertIn('training result', local)
                self.assertIn('training result', shared)
                self.assertIn('only in local crash log', local)
                self.assertNotIn('only in local crash log', shared)
                self.assertIn('Log End', local)
                self.assertFalse(running.exists())
            self.assertFalse(crash.filename.exists())
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(copy.call_args.args[0], crash.filename)

    def test_deleted_shared_file_does_not_break_local_archive(self):
        with CrashProtectorCatcher('task') as crash:
            with MarkdownCatcher('training', to_share_folder=True, crash_protector=crash) as md:
                md.running_filename.unlink()
                print('result after shared deletion')
                old_shared = md.share_filename
                old_shared.write_text('previous archive')
            self.assertIn('result after shared deletion', md.export_file_list[0].read_text())
            self.assertEqual(old_shared.read_text(), 'previous archive')
        self.assertIn('result after shared deletion', md.get_contents())

    def test_failed_local_copy_preserves_previous_archive_and_crash_log(self):
        with CrashProtectorCatcher('task') as crash:
            with MarkdownCatcher('training', to_share_folder=True, crash_protector=crash) as md:
                print('recoverable result')
                target = md.export_file_list[0]
                target.parent.mkdir(parents=True)
                target.write_text('previous archive')
                original_copy = shutil.copyfile

                def partial_copy(source, destination):
                    Path(destination).write_text('incomplete')
                    raise OSError('simulated disk failure')

                failed = patch('src.proj.util.catcher.crash_protector.shutil.copyfile', side_effect=partial_copy)
                failed.start()
                self.addCleanup(failed.stop)
            failed.stop()
            self.assertEqual(shutil.copyfile, original_copy)
            self.assertEqual(target.read_text(), 'previous archive')
            self.assertEqual(list(target.parent.glob('*.tmp')), [])
            self.assertIn('recoverable result', md.share_filename.read_text())
        self.assertTrue(crash.filename.exists())
        self.assertTrue(crash.markdown_file.closed)
        self.assertIn('recoverable result', crash.filename.read_text())

    def test_same_title_catchers_do_not_delete_each_others_running_files(self):
        with MarkdownCatcher('same title', to_share_folder=True) as outer:
            print('outer before')
            with MarkdownCatcher('same title', to_share_folder=True) as inner:
                print('inner')
                self.assertNotEqual(outer.running_filename, inner.running_filename)
            self.assertTrue(outer.running_filename.exists())
            print('outer after')
        self.assertIn('outer after', outer.share_filename.read_text())
        self.assertIn('outer after', outer.export_file_list[0].read_text())

    def test_standalone_without_share_folder_uses_local_crash_log(self):
        with MarkdownCatcher('local only', to_share_folder=False) as md:
            print('local result')
            self.assertTrue(md.crash_protector.filename.exists())
            self.assertIsNone(md.running_filename)
        self.assertIn('local result', md.get_contents())
        self.assertFalse(md.crash_protector.filename.exists())

    def test_unavailable_share_folder_does_not_break_local_archive(self):
        PATH.share_folder.write_text('not a directory')
        with MarkdownCatcher('unavailable share', to_share_folder=True) as md:
            print('still captured locally')
        self.assertIn('still captured locally', md.get_contents())

    def test_shared_write_failure_does_not_interrupt_local_logging(self):
        with MarkdownCatcher('write failure', to_share_folder=True) as md:
            with patch.object(md.markdown_writer, 'write', side_effect=OSError('shared disk unavailable')):
                print('first local result')
                print('second local result')
        self.assertIn('first local result', md.get_contents())
        self.assertIn('second local result', md.get_contents())

    def test_autorun_reuses_active_crash_protector_and_handles_no_task_id(self):
        for task_id in ['autorun_task', None]:
            with self.subTest(task_id=task_id):
                catchers = AutoRunCatchers(task_id, warning_catcher=False)
                catchers.enter(f'task {task_id}', 'test', datetime.now())
                try:
                    crash, html, md = catchers.catchers
                    if task_id is not None:
                        self.assertIs(md.crash_protector, crash)
                    print(f'result for {task_id}')
                finally:
                    catchers.exit(None, None, None)
                self.assertIn(f'result for {task_id}', md.get_contents())
                self.assertFalse(md.crash_protector.filename.exists())
                self.assertTrue(html.export_file_list[0].exists())

    def test_no_false_success_message_for_failed_archive(self):
        with MarkdownCatcher('failed archive', to_share_folder=False) as md:
            print('keep this log')
            success = self.enterContext(patch.object(md.logger, 'footnote'))
            self.enterContext(patch('src.proj.util.catcher.crash_protector.shutil.copyfile',
                                    side_effect=OSError('simulated copy failure')))
        self.assertFalse(any('result saved to' in str(call) for call in success.call_args_list))
        self.assertTrue(md.crash_protector.filename.exists())

    def test_task_exception_propagates_after_archive(self):
        with self.assertRaisesRegex(RuntimeError, 'task failed'):
            with MarkdownCatcher('exception', to_share_folder=True) as md:
                print('output before exception')
                raise RuntimeError('task failed')
        self.assertIn('output before exception', md.get_contents())
        self.assertFalse(md.crash_protector.filename.exists())


if __name__ == '__main__':
    unittest.main()
