"""Remote transport regressions use private temporary shares and no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.task_monitor.remote import Publisher, CHUNK_BYTES, RETENTION, read_volume, retain_log, task_key
from tests.task_monitor.test_core import _create_database


class RemotePublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / 'source.md'
        self.publisher = Publisher(self.root / 'shared', self.root / 'private', 'server')
        self.task = dict(task_id='run-1', script='train.py', status='running', log=str(self.log))

    def tick(self, now=100):
        return self.publisher.tick([self.task], now=now)['tasks'][0]

    def test_idle_schedule_log_marker_deduplicates_without_a_database_link(self):
        from src.api.task_monitor.core import TaskMonitorRepository
        from src.api.task_monitor.remote import collect
        from src.api.task_monitor.web_api import TaskCenter
        from src.automation.scheduling.runtime import RunStore
        from tests.task_monitor.test_core import _create_database

        runtime = self.root / 'runtime'
        database = self.root / 'task.sqlite'
        store = RunStore(runtime / 'scheduling' / 'runs.sqlite')
        script_id = '4_train/2_schedule_model.py@100'
        wrapper_id = 'managed:scheduler:scheduled'
        _create_database(database, [(script_id, '4_train/2_schedule_model.py', '', 100, 'running', 'bash',
                                     None, 105, None, None, None, None)])
        captures = runtime / 'crash_protector'
        captures.mkdir()
        capture = captures / '20261124.4_train_2_schedule_model.py@100.md'
        capture.write_text('training output\n')
        self.log.write_text(f'\x1b[94mCrashProtectorCatcher >> task_id="{script_id}", Capturing Start\x1b[0m\n')
        run = dict(id='scheduled', task='factorseq_graph', owner='idle_worklist', phase='running',
                   queued_at=100, started_at=103, log_path=str(self.log))
        store.put(run)
        repository = TaskMonitorRepository(database, runtime)
        hidden = set()
        tasks = collect(repository, now=110, hidden=hidden)
        self.assertEqual(hidden, {wrapper_id})
        self.assertEqual([task['task_id'] for task in tasks], [script_id])

        # A client with a new viewer and an old server snapshot can also identify the wrapper.
        wrapper = dict(task_id=wrapper_id, script='factorseq_graph', status='running',
                       cmd='watchdog / idle_worklist / scheduler', log=str(self.log))
        standalone = dict(wrapper, task_id='managed:scheduler:standalone', log=None)
        self.publisher.tick([*tasks, wrapper, standalone], now=110)
        center = TaskCenter(repository=repository, shared_root=self.root / 'shared')
        remote = center.remote_list()[0]
        self.assertEqual({task['task_id'] for task in remote['tasks']},
                         {script_id, 'managed:scheduler:standalone'})
        first_segment = next((self.publisher.shared / 'tasks' / task_key(script_id)).glob('*.bin'))
        first_segment.unlink()
        center.remote_cache.clear_task(self.publisher.shared, script_id)
        self.assertIn(wrapper_id, {row['task_id'] for row in center.remote_list()[0]['tasks']})

    def test_sealed_volumes_are_immutable_and_restart_resumes(self):
        self.log.write_bytes(b'a' * (CHUNK_BYTES + 13))
        task = self.tick()
        sealed = self.publisher.shared / 'tasks' / task_key('run-1') / task['segments'][0]['file']
        stamp = sealed.stat().st_mtime_ns
        with self.log.open('ab') as output:
            output.write(b'last line\n')
        self.publisher = Publisher(self.root / 'shared', self.root / 'private', 'server')
        task = self.tick(115)
        self.assertEqual(sealed.stat().st_mtime_ns, stamp)
        self.assertEqual(task['offset'], self.log.stat().st_size)
        actual = self.volume(task, 0) + self.volume(task, 1)
        self.assertEqual(actual.encode(), self.log.read_bytes())

    def test_short_task_handoff_and_cleanup_preserves_other_task(self):
        from types import SimpleNamespace
        self.log.write_text('finished\n')
        config = SimpleNamespace(state=self.root / 'private')
        with patch('src.api.task_monitor.remote.settings', return_value=(config, True)):
            self.assertTrue(retain_log(self.log, 'run-1'))
        self.assertFalse(self.log.exists())
        self.task['status'] = 'complete'
        task = self.tick()
        self.assertEqual(read_volume(self.publisher.shared, task, 0), 'finished\n')
        self.assertEqual(task['expires_at'], 100 + RETENTION)
        other = dict(self.task, task_id='run-2', status='running', log=None)
        packet = self.publisher.tick([other], now=100 + RETENTION)
        self.assertEqual([t['task_id'] for t in packet['tasks']], ['run-2'])
        self.assertFalse((self.publisher.shared / 'tasks' / task_key('run-1')).exists())
        self.assertFalse((self.root / 'private/remote_monitor/pending' / (task_key('run-1') + '.md')).exists())

    def test_published_queue_entries_disappear_when_linked_or_deferred(self):
        from src.api.task_monitor.core import TaskMonitorRepository
        from src.api.task_monitor.remote import collect
        from src.automation.scheduling.runtime import RunStore

        runtime = self.root / 'runtime'
        store = RunStore(runtime / 'scheduling' / 'runs.sqlite')
        db = self.root / 'tasks.sqlite'
        repository = TaskMonitorRepository(db, runtime)
        linked = {'id': 'linked', 'phase': 'queued', 'task': 'model', 'queued_at': 100}
        deferred = dict(linked, id='deferred')
        queued = dict(linked, id='queued')
        for run in (linked, deferred, queued):
            store.put(run)
        packet = self.publisher.tick(collect(repository, now=100), now=100)
        known = {row['task_id'] for row in packet['tasks']}
        self.assertEqual(len(known), 3)

        _create_database(db, [('script', 'schedule.py', '', 111, 'running', 'bash',
                              123, 111, None, None, None, None)])
        store.put(dict(linked, phase='running', pid=123, started_at=110))
        store.link('script', 'linked')
        store.put(dict(deferred, phase='deferred'))
        hidden = set()
        tasks = collect(repository, now=115, pending_ids=known, hidden=hidden)
        self.assertEqual(hidden, {'managed:scheduler:linked', 'managed:scheduler:deferred'})
        self.publisher.tick(tasks, now=115, hidden=hidden)
        index = json.loads((self.publisher.shared / 'index.json').read_text())
        self.assertEqual({row['task_id'] for row in index['tasks']}, {'script', 'managed:scheduler:queued'})

        packet = self.publisher.tick([], now=130)
        self.assertEqual({row['task_id'] for row in packet['tasks']}, {'script', 'managed:scheduler:queued'})
        self.publisher.tick([], now=115 + RETENTION)
        self.assertFalse((self.publisher.shared / 'tasks' / task_key('managed:scheduler:linked')).exists())

    def test_replacement_retains_previous_history_and_missing_volume_waits(self):
        self.log.write_text('before\n')
        self.tick()
        self.log.unlink()
        self.log.write_text('after\n')
        task = self.tick(115)
        self.assertTrue(task['gap'])
        self.assertEqual(task['previous'][0]['offset'], 7)
        volume = self.publisher.shared / 'tasks' / task_key('run-1') / task['segments'][-1]['file']
        volume.unlink()
        self.assertIsNone(read_volume(self.publisher.shared, task, 0))

    def test_utf8_crosses_volume_boundary(self):
        text = 'a' * (CHUNK_BYTES - 1) + '中文\n'
        self.log.write_text(text)
        task = self.tick()
        actual = self.volume(task, 0) + self.volume(task, 1)
        self.assertEqual(actual, text)

    def volume(self, task, number):
        content = read_volume(self.publisher.shared, task, number)
        assert content is not None
        return content

    def test_close_hands_off_once_even_when_both_deflectors_close(self):
        from types import SimpleNamespace
        from src.proj.util.catcher.crash_protector import CrashProtectorCatcher
        output = self.log.open('w')
        output.write('last output\n')
        catcher = object.__new__(CrashProtectorCatcher)
        catcher.markdown_file = output
        catcher.filename = self.log
        catcher.task_id = 'run-1'
        catcher.retain_file = False
        with patch('src.api.task_monitor.remote.settings', return_value=(SimpleNamespace(state=self.root / 'private'), True)):
            catcher.close()
            catcher.close()
        self.assertFalse(catcher.retain_file)
        pending = self.root / 'private/remote_monitor/pending' / (task_key('run-1') + '.md')
        self.assertEqual(pending.read_text(), 'last output\n')

    def test_catchup_budget_and_retry_after_cursor_write_failure(self):
        from src.api.task_monitor import remote_transport as transport
        self.log.write_bytes(b'x' * (transport.ROUND_BYTES + 41))
        original = transport.atomic_json
        def fail_cursor(path, value):
            if path.name == 'state.json':
                raise OSError('interrupted before cursor commit')
            original(path, value)
        with patch.object(transport, 'atomic_json', side_effect=fail_cursor):
            with self.assertRaises(OSError):
                self.tick()
        task = self.tick(115)
        self.assertEqual(task['offset'], self.log.stat().st_size)
        self.assertTrue(task['complete'])
        task = self.tick(130)
        self.assertTrue(task['complete'])
        chunks = (self.volume(task, number)
                  for number in range((task['offset'] + CHUNK_BYTES - 1) // CHUNK_BYTES))
        self.assertEqual(''.join(chunks).encode(), self.log.read_bytes())

    def test_old_pending_task_is_collected_after_long_outage(self):
        from src.api.task_monitor.remote import collect
        from src.api.task_monitor.core import TaskMonitorRepository
        database = self.root / 'tasks.sqlite'
        _create_database(database, [('run-1', 'train.py', 'python train.py', 1, 'complete',
                                     'bash', None, 1, 2, 0, None, None)])
        repository = TaskMonitorRepository(database, self.root / 'runtime', self.root / 'autofix.sqlite')
        self.assertEqual(collect(repository, now=2 * RETENTION), [])
        tasks = collect(repository, now=2 * RETENTION, pending_ids=['run-1'])
        self.assertEqual(tasks[0]['status'], 'complete')

    def test_idle_does_not_rewrite_manifests_or_logs(self):
        self.log.write_bytes(b'a')
        self.tick()
        paths = [self.publisher.shared / 'index.json',
                 self.publisher.shared / 'tasks' / task_key('run-1') / 'manifest.json']
        paths += list(self.publisher.shared.rglob('*.bin'))
        stamps = {path: path.stat().st_mtime_ns for path in paths}
        self.tick(115)
        self.assertEqual(stamps, {path: path.stat().st_mtime_ns for path in paths})

    def test_two_stage_merge_retains_three_raw_fragments_and_cached_bytes(self):
        from src.api.task_monitor.remote_reader import RemoteCache
        cache = RemoteCache()
        self.log.write_bytes(b'')
        for index in range(259):
            with self.log.open('ab') as output:
                output.write(bytes([index % 251]) * 1024)
            task = self.tick(100 + index * 15)
            if index == 66:
                self.assertEqual([s['kind'] for s in task['segments']].count('intermediate'), 1)
                self.assertEqual(cache.read(self.publisher.shared, task, 0, 64 * 1024), self.log.read_bytes()[:64 * 1024])
        kinds = [s['kind'] for s in task['segments']]
        self.assertEqual(kinds, ['sealed', 'raw', 'raw', 'raw'])
        self.assertEqual(task['segments'][0]['end'], CHUNK_BYTES)
        with patch.object(Path, 'open', side_effect=AssertionError('Cached range was read again')):
            self.assertEqual(cache.read(self.publisher.shared, task, 0, 64 * 1024), b''.join(bytes([i % 251]) * 1024 for i in range(64)))
        fresh = RemoteCache()
        self.assertEqual(fresh.read(self.publisher.shared, task, 0, task['offset']), self.log.read_bytes())

    def test_incomplete_transport_recovers_when_replacement_arrives(self):
        from src.api.task_monitor.remote_reader import RemoteCache
        self.log.write_bytes(b'')
        for index in range(67):
            with self.log.open('ab') as output:
                output.write(b'x')
            task = self.tick(100 + index * 15)
        merged = self.publisher.shared / 'tasks' / task_key('run-1') / task['segments'][0]['file']
        payload = merged.read_bytes()
        merged.unlink()
        cache = RemoteCache()
        self.assertIsNone(cache.read(self.publisher.shared, task, 0, task['offset']))
        merged.write_bytes(payload)
        self.assertEqual(cache.read(self.publisher.shared, task, 0, task['offset']), b'x' * 67)

    def test_v1_partial_volume_is_adopted_without_modification(self):
        self.log.write_bytes(b'old')
        directory = self.publisher.shared / 'tasks' / task_key('run-1')
        directory.mkdir(parents=True)
        volume = directory / '0-00000000.bin'
        volume.write_bytes(b'old')
        self.publisher.private.mkdir(parents=True)
        stat = self.log.stat()
        state = dict(cursor=0, tasks={task_key('run-1'): dict(task=self.task, offset=3, generation=0,
                                                           identity=[stat.st_dev, stat.st_ino])})
        (self.publisher.private / 'state.json').write_text(json.dumps(state))
        stamp = volume.stat().st_mtime_ns
        with self.log.open('ab') as output:
            output.write(b'new')
        task = self.tick()
        self.assertEqual(read_volume(self.publisher.shared, task, 0), 'oldnew')
        self.assertEqual(volume.stat().st_mtime_ns, stamp)

    def test_merge_interruption_recovers_from_private_journal(self):
        self.log.write_bytes(b'')
        for index in range(66):
            with self.log.open('ab') as output:
                output.write(b'x')
            self.tick(100 + index * 15)
        with self.log.open('ab') as output:
            output.write(b'x')
        original = Path.unlink
        failed = False
        def interrupt(path, *args, **kwargs):
            nonlocal failed
            if path.suffix == '.bin' and not failed:
                failed = True
                raise OSError('interrupted cleanup')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'unlink', interrupt):
            with self.assertRaises(OSError):
                self.tick(1200)
        self.publisher = Publisher(self.root / 'shared', self.root / 'private', 'server')
        task = self.tick(1215)
        self.assertEqual(read_volume(self.publisher.shared, task, 0), 'x' * 67)
        self.assertFalse((self.publisher.private / 'journal.json').exists())

    def test_render_cache_survives_automatic_refresh_and_has_memory_limit(self):
        from src.api.task_monitor.remote_reader import RemoteCache
        self.log.write_text('cached output\n')
        task = self.tick()
        cache = RemoteCache(max_bytes=64)
        first = cache.render(self.publisher.shared, task, 0, task['offset'])
        with patch.object(Path, 'open', side_effect=AssertionError('Unexpected history read')), \
             patch('src.api.task_monitor.output.sanitize_inline_html', side_effect=AssertionError('Unexpected parsing')):
            self.assertEqual(cache.render(self.publisher.shared, task, 0, task['offset']), first)
        cache.clear_task(self.publisher.shared, task['task_id'])
        self.assertEqual(cache.total, 0)
        with self.log.open('ab') as output:
            output.write(b'x' * 100)
        task = self.tick(115)
        cache.render(self.publisher.shared, task, 0, task['offset'])
        self.assertLessEqual(cache.total, cache.max_bytes)

    def test_metadata_is_parsed_only_when_file_changes(self):
        from src.api.task_monitor.remote_reader import RemoteCache
        self.log.write_text('metadata\n')
        self.tick()
        path = self.publisher.shared / 'index.json'
        cache = RemoteCache()
        original = cache.document(path)
        with patch.object(Path, 'read_text', side_effect=AssertionError('Unexpected index read')):
            self.assertEqual(cache.document(path), original)
        self.task['status'] = 'complete'
        self.tick(115)
        updated = cache.document(path)
        assert updated is not None
        self.assertEqual(updated['tasks'][0]['status'], 'complete')

    def test_idle_and_merge_payload_write_budget(self):
        self.log.write_bytes(b'')
        payload_bytes = 0
        original = self.publisher._finish
        def count(journal):
            nonlocal payload_bytes
            payload_bytes += sum((self.publisher.private / operation['source']).stat().st_size
                                 for operation in journal['writes'] if operation['immutable'])
            return original(journal)
        with patch.object(self.publisher, '_finish', side_effect=count):
            for index in range(259):
                with self.log.open('ab') as output:
                    output.write(b'x' * 1024)
                self.tick(100 + index * 15)
            before_idle = payload_bytes
            self.tick(5000)
        self.assertEqual(payload_bytes, before_idle)
        # Raw publication plus 64-fragment packing plus the final 256 KiB merge.
        self.assertEqual(payload_bytes, (259 + 256 + 256) * 1024)
        # The former mutable-tail protocol rewrote the accumulated partial volume each round.
        old_payload_bytes = (sum(range(1, 257)) + sum(range(1, 4))) * 1024
        self.assertLess(payload_bytes * 40, old_payload_bytes)

    def test_worker_drains_restart_backlog_without_waiting_for_next_timer(self):
        from types import SimpleNamespace
        from src.api.task_monitor.remote import publish_once, ROUND_BYTES
        self.log.write_bytes(b'first')
        self.tick()
        with self.log.open('ab') as output:
            output.write(b'z' * (ROUND_BYTES + 1024))
        config = SimpleNamespace(state=self.root / 'private', shared_root=self.root / 'shared', source_host='server')
        repository = SimpleNamespace(runtime_dir=self.root / 'runtime')
        with patch('src.api.task_monitor.remote.settings', return_value=(config, True)), \
             patch('src.api.task_monitor.core.TaskMonitorRepository', return_value=repository), \
             patch('src.api.task_monitor.remote.collect', return_value=[self.task]), \
             patch('src.automation.scheduling.worker_bootstrap.wait_for_code', return_value=False):
            self.assertEqual(publish_once(), 0)
        state = json.loads((self.publisher.private / 'state.json').read_text())
        record = state['tasks'][task_key('run-1')]
        self.assertEqual(record['offset'], self.log.stat().st_size)
        self.assertTrue(record['complete'])
