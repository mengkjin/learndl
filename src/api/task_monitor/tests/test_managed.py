import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from src.api.task_monitor.core import TaskMonitorRepository
from src.api.task_monitor.actions import kill_unavailable_reason
from src.api.task_monitor.tests.test_core import _create_database


class ManagedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.db=self.root/'tasks.db';self.runs=self.root/'scheduling/runs.sqlite'
        self.runs.parent.mkdir()
        with sqlite3.connect(self.runs) as db:
            db.executescript('CREATE TABLE runs(id TEXT, data TEXT); CREATE TABLE task_links(task_id TEXT,run_id TEXT);')
        self.repo=TaskMonitorRepository(self.db,self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self,key,**kwargs):
        doc=dict(id=key,task='factor_sequence:'+key,owner='idle_worklist',phase='running',queued_at=1,started_at=2,**kwargs)
        with sqlite3.connect(self.runs) as db:
            db.execute('INSERT INTO runs VALUES (?,?)',(key,json.dumps(doc)))

    def test_missing_script_database_still_shows_managed(self):
        self.add('long',pid=12,process_created=1,log_path=str(self.root/'worker.log'))
        proc=Mock();proc.create_time.return_value=999
        with patch('src.api.task_monitor.managed.psutil.Process',return_value=proc):
            page=self.repo.list_tasks(window='last_day',now=200000)
            self.assertEqual(page.total,1)
            self.assertEqual(page.tasks[0].display_status,'stale')
            task=self.repo.get_task(page.tasks[0].task_id)
        self.assertEqual(task.exit_files,(self.root/'worker.log',))
        self.assertTrue(task.is_background)
        self.assertIsNotNone(kill_unavailable_reason(task))
        self.assertEqual(self.repo.status_counts(window='last_day',now=200000)['running'],1)
        self.assertFalse(self.db.exists())

    def test_union_pagination_counts_and_same_process_dedup(self):
        _create_database(self.db,[('linked','x.py','x',1,'running','script',12,2,None,None,None,None),
                                 ('child','y.py','y',3,'running','script',13,3,None,None,None,None)])
        self.add('parent',pid=12)
        self.add('unwrapped',pid=14)
        with sqlite3.connect(self.runs) as db:
            db.execute('INSERT INTO task_links VALUES (?,?)',('linked','parent'))
            db.execute('INSERT INTO task_links VALUES (?,?)',('child','unwrapped'))
        first=self.repo.list_tasks(window='history',limit=1)
        second=self.repo.list_tasks(window='history',limit=1,cursor=first.next_cursor)
        third=self.repo.list_tasks(window='history',limit=1,cursor=second.next_cursor)
        ids=[p.tasks[0].task_id for p in (first,second,third)]
        self.assertEqual(len(set(ids)),3)
        self.assertIn('managed:scheduler:unwrapped',ids)
        self.assertNotIn('managed:scheduler:parent',ids)
        self.assertIsNone(third.next_cursor)
        self.assertEqual(first.total,3)
        self.assertEqual(self.repo.status_counts(window='history')['running'],3)

    def test_autofix_history_and_corrupt_row_are_readonly(self):
        af=self.root/'ledger.sqlite'
        with sqlite3.connect(af) as db:
            db.execute('CREATE TABLE fixes(data TEXT)')
            db.execute('INSERT INTO fixes VALUES (?)',(json.dumps({'id':'fix','phase':'failed','started_at':10,
                'finished_at':20,'item':{'script':'repair.py'},'error':'bad'}),))
            db.execute('INSERT INTO fixes VALUES (?)',('broken',))
        before=af.read_bytes()
        repo=TaskMonitorRepository(self.db,self.root,af)
        page=repo.list_tasks(statuses={'error'},window='history')
        self.assertEqual(page.total,1)
        self.assertEqual(repo.get_task(page.tasks[0].task_id).exit_error,'bad')
        self.assertEqual(repo.status_counts(window='history')['error'],1)
        self.assertEqual(before,af.read_bytes())
        self.assertEqual(repo.list_tasks(statuses=set(),window='history').total,0)


if __name__=='__main__':
    unittest.main()
