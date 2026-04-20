"""
Task lifecycle management for the interactive Streamlit application.

Classes
-------
TaskDatabase
    SQLite-backed persistence layer for all task, queue, and exit-file records.
TaskQueue
    In-memory ordered collection of :class:`TaskItem` objects backed by
    :class:`TaskDatabase`, with filtering, pagination, and refresh logic.
TaskItem
    Dataclass representing a single pipeline script execution, including its
    process PID, status transitions, timing, and output metadata.

Module-level helpers
--------------------
timestamp()
    Current POSIX timestamp as a float.
runs_page_url(script_key)
    Derive the Streamlit page URL for a given script key.
"""

from __future__ import annotations
import os , sys , re , time
import pandas as pd
from typing import Any , Literal , Sequence
from dataclasses import dataclass , field , asdict
from datetime import datetime
from pathlib import Path

from src.proj import PATH , Logger , Duration
from src.proj.util import ScriptCmd , DBConnHandler , Email , properties , check_process_status , kill_process

def timestamp() -> float:
    """Return the current UTC time as a POSIX timestamp (seconds since epoch)."""
    return datetime.now().timestamp()

def runs_page_url(script_key : str):
    """get runs page url"""
    return "pages/_" + re.sub(r'[/\\]', '_', script_key)

class TaskDatabase:
    """SQLite-backed persistence layer for task and queue records.

    Tables managed
    --------------
    task_records
        One row per task with status, timing, PID, and exit metadata.
    task_exit_files
        Zero-or-more output file paths per task.
    task_backend_updated
        Tracks tasks that have been updated by the backend process and need a
        frontend refresh.
    queue_records
        Registry of named queues; the most-recently created is the active queue.
    task_queues
        Many-to-many mapping of queues to task IDs.
    """
    def __init__(self , db_name: str | Path | None = None) -> None:
        """Initialise the database connection and create tables if they don't exist.

        Parameters
        ----------
        db_name:
            Override the default database file stem. Useful for tests or
            isolated sessions.
        """
        self.db_path = self.get_db_path()
        if db_name is not None:
            self.db_path = self.db_path.with_name(f'{db_name}.db')
        self.db_name = self.db_path.stem
        self.conn_handler = DBConnHandler(self.db_path)
        self.initialize_database()

    @staticmethod
    def get_db_path() -> Path:
        """Return the default path to the SQLite database file."""
        return PATH.app_db / 'interactive_tasks.db'

    def initialize_database(self) -> None:
        """Initialize database and tables, creating them if they do not already exist."""
        # create 5 main tables : 
        # task_records : task records
        # task_exit_files : task exit files
        # task_backend_updated : task backend updated records
        # queue_records : queue records , last one is the active queue
        # task_queues : task queues and their task_ids
            
        with self.conn_handler(check_same_thread = True) as (conn, cursor):
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_records (
                    task_id TEXT PRIMARY KEY,
                    script TEXT NOT NULL,
                    cmd TEXT NOT NULL,
                    create_time REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT,
                    pid INTEGER,
                    start_time REAL,
                    end_time REAL,
                    exit_code INTEGER,
                    exit_message TEXT,
                    exit_error TEXT
                )
                ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS ix_task_records_script ON task_records(script)')
            cursor.execute('CREATE INDEX IF NOT EXISTS ix_task_records_status ON task_records(status)')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_exit_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES task_records(task_id)
                )
                ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_backend_updated (
                    task_id TEXT NOT NULL,
                    updated_time REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES task_records(task_id)
                )
                ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queue_records (
                    queue_id TEXT NOT NULL PRIMARY KEY,
                    create_time REAL NOT NULL
                )
                ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_queues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    FOREIGN KEY (queue_id) REFERENCES queue_records(queue_id),
                    FOREIGN KEY (task_id) REFERENCES task_records(task_id)
                )
                ''')
            
    @property
    def empty(self) -> bool:
        """True if the task_records table contains no rows."""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_records')
            return cursor.fetchone() is None
        
    @classmethod
    def backup_stats(cls , backup_path : Path | str):
        """Get task count from backup database"""
        with DBConnHandler(backup_path) as (conn, cursor):
            cursor.execute('SELECT COUNT(*) FROM task_records')
            task_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM queue_records')
            queue_count = cursor.fetchone()[0]
            return {
                'task_count': task_count,
                'queue_count': queue_count,
            }

    def task_count(self) -> int:
        """Return the total number of task records in the database."""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT COUNT(*) FROM task_records')
            return cursor.fetchone()[0]
            
    def clear_database(self) -> None:
        """Clear all records from the database, backing up first to prevent data loss."""
        self.conn_handler.backup()
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_records')
            cursor.execute('DELETE FROM task_exit_files')
            cursor.execute('DELETE FROM queue_records')
            cursor.execute('DELETE FROM task_queues')
            cursor.execute('DELETE FROM task_backend_updated')
            cursor.execute('DELETE FROM sqlite_sequence')
        self.initialize_database()
    
    def new_task(self, task: TaskItem , overwrite: bool = False) -> None:
        """Insert a new task record (and its exit-file rows) into the database.

        Parameters
        ----------
        task:
            The :class:`TaskItem` to persist.
        overwrite:
            If True, delete any existing record with the same ``task_id`` first.
        """
        with self.conn_handler as (conn, cursor):
            if overwrite:
                cursor.execute('DELETE FROM task_records WHERE task_id = ?', (task.id,))
                
            cursor.execute('''
            INSERT INTO task_records (
                task_id, script, cmd, create_time, 
                status, source, pid, start_time, end_time, 
                exit_code, exit_message, exit_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id,
                task.script,
                task.cmd,
                task.create_time,
                task.status,
                task.source,
                task.pid,
                task.start_time,
                task.end_time,
                task.exit_code,
                task.exit_message,
                task.exit_error
            ))

            cursor.execute("DELETE FROM task_exit_files WHERE task_id = ?", (task.id,))
            for file_path in task.exit_files or []:
                cursor.execute('''
                INSERT INTO task_exit_files (task_id, file_path)
                VALUES (?, ?)
                ''', (task.id, file_path))

    def new_queue(self, queue_id: str , exist_ok: bool = False) -> None:
        """Register a new queue in queue_records.

        Parameters
        ----------
        queue_id:
            Unique identifier for the queue.
        exist_ok:
            If True, silently skip insertion when the queue already exists.
        """
        with self.conn_handler as (conn, cursor):
            if exist_ok:
                cursor.execute('SELECT * FROM queue_records WHERE queue_id = ?', (queue_id,))
                if cursor.fetchone() is not None: 
                    return
            cursor.execute('''
            INSERT INTO queue_records (queue_id, create_time)
                VALUES (?, ?)
                ''', (queue_id, timestamp()))
    
    def update_task(self, task_id: str, backend_updated: bool = False, **kwargs) -> None:
        """Update one or more columns of an existing task record.

        Parameters
        ----------
        task_id:
            The task to update.
        backend_updated:
            If True, upsert ``task_backend_updated`` so the frontend knows to
            refresh this task on the next poll.
        **kwargs:
            Column-value pairs to write; ``exit_files`` is handled separately.
        """
        if not kwargs or task_id == '': 
            return
        with self.conn_handler as (conn, cursor):
            exit_files = kwargs.pop('exit_files' , None)
            query = ' '.join([
                "UPDATE task_records SET",
                ", ".join([f"{k} = ?" for k in kwargs.keys()]),
                "WHERE task_id = ?"
            ])
            params = list(kwargs.values()) + [task_id]
            cursor.execute(query, params)
            
            if exit_files:
                cursor.execute("DELETE FROM task_exit_files WHERE task_id = ?", (task_id,))
                for file_path in exit_files:
                    cursor.execute('''
                    INSERT INTO task_exit_files (task_id, file_path)
                    VALUES (?, ?)
                    ''', (task_id, file_path))
            if backend_updated:
                cursor.execute(
                    '''
                    INSERT INTO task_backend_updated (task_id, updated_time)
                    VALUES (?, ?)
                    ''',
                    (task_id, timestamp()),
                )

        new_status = self.check_task_status(task_id)
        if 'status' in kwargs:
            if kwargs['status'] != new_status:
                raise ValueError(f"Task {task_id} status update failed , expected {kwargs['status']} but got {new_status}")
            else:
                new_kwargs = {'status' : new_status} | {k: v for k, v in kwargs.items() if k != 'status'}
                for k, v in new_kwargs.items():
                    if k.endswith('_time'):
                        assert v is None or isinstance(v, float) , f'{k} must be a float, but got {type(v)}'
                        new_kwargs[k] = f'{v} ({datetime.fromtimestamp(v).strftime('%Y-%m-%d %H:%M:%S')})' if v else 'None'
                Logger.info(f"Task {task_id} status [{new_status.upper()}]")

    def check_task_status(self, task_id: str) -> Literal['starting', 'running', 'complete', 'error' , 'killed']:
        """Check task status"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT status FROM task_records WHERE task_id = ?', (task_id,))
            return cursor.fetchone()['status']

    def get_backend_updated_tasks(self) -> list[str]:
        """Get backend updated tasks"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT task_id FROM task_backend_updated')
            return [row['task_id'] for row in cursor.fetchall()]
        
    def clear_backend_updated_tasks(self, task_ids: list[str] | None = None) -> None:
        """Remove all entries from the task_backend_updated tracking table."""
        if not task_ids:
            return
        ph = ','.join('?' * len(task_ids))
        with self.conn_handler as (conn, cursor):
            cursor.execute(f'DELETE FROM task_backend_updated WHERE task_id IN ({ph})', tuple(task_ids),)

    def is_backend_updated(self, task_id: str) -> bool:
        """Check if task is backend updated"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_backend_updated WHERE task_id = ?', (task_id,))
            return cursor.fetchone() is not None
        
    def del_backend_updated_task(self, task_id: str) -> None:
        """Remove the backend-updated marker for a single task."""
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_backend_updated WHERE task_id = ?', (task_id,))

    def sync_queue(self, queue_id: str) -> None:
        """Rebuild the task_queues entries for *queue_id* from all task_records."""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT task_id FROM task_records')
            task_ids = [row['task_id'] for row in cursor.fetchall()]
            cursor.execute('DELETE FROM task_queues WHERE queue_id = ?', (queue_id,))
            for task_id in task_ids:
                cursor.execute('INSERT INTO task_queues (queue_id, task_id) VALUES (?, ?)', (queue_id, task_id))

    def clear_queue(self, queue_id: str) -> None:
        """Remove all task associations for the given queue (task records are kept)."""
        with self.conn_handler as (conn, cursor):
            cursor.execute("DELETE FROM task_queues WHERE queue_id = ?", (queue_id,))
    
    def get_task(self, task_id: str) -> TaskItem | None:
        """Get task information"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_records WHERE task_id = ?', (task_id,))
            task = cursor.fetchone()
            if not task: 
                return None

            cursor.execute('SELECT file_path FROM task_exit_files WHERE task_id = ?', (task_id,))
            files = [row['file_path'] for row in cursor.fetchall()]

            task = dict(task) | {'exit_files': files}
            item = TaskItem(**task)
        return item
    
    def get_tasks(self, task_ids: list[str]) -> list[TaskItem]:
        """Get tasks information"""
        tasks = []
        with self.conn_handler as (conn, cursor):
            placeholders = ','.join(['?'] * len(task_ids))
            query = f'SELECT * FROM task_records WHERE task_id IN ({placeholders})'
            cursor.execute(query, task_ids)
            tasks = {row['task_id']: TaskItem(**dict(row)) for row in cursor.fetchall()}

            query = f'SELECT * FROM task_exit_files WHERE task_id IN ({placeholders})'
            cursor.execute(query, task_ids)
            exit_files : dict[str, list[str]] = {}
            for row in cursor.fetchall():
                if row['task_id'] not in exit_files:
                    exit_files[row['task_id']] = []
                exit_files[row['task_id']].append(row['file_path'])
            for task_id, files in exit_files.items():
                tasks[task_id].exit_files = files or None

        return list(tasks.values())
    
    def add_queue_task(self, queue_id: str , task_id: str) -> None:
        """Associate a task with a queue in task_queues."""
        with self.conn_handler as (conn, cursor):
            cursor.execute('INSERT INTO task_queues (queue_id, task_id) VALUES (?, ?)', (queue_id, task_id))
    
    def del_queue_task(self, queue_id: str , task_id: str) -> None:
        """Remove a task association from a queue."""
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_queues WHERE queue_id = ? AND task_id = ?', (queue_id, task_id))

    def get_queue_tasks(self, queue_id: str , max_queue_size: int | None = None) -> list[TaskItem]:
        """Get queue information"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT task_id FROM task_queues WHERE queue_id = ?', (queue_id,))
            task_ids = [row['task_id'] for row in cursor.fetchall()]
            if max_queue_size: 
                task_ids = task_ids[-max_queue_size:]
            return self.get_tasks(task_ids)
    
    def active_queue(self) -> str:
        '''get the last queue in table queue_records'''
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT queue_id FROM queue_records ORDER BY create_time DESC LIMIT 1')
            queue = cursor.fetchone()
            if queue: 
                return queue['queue_id']
            else:
                return 'default'
    
    def del_task(self, task_id: str , check = True , force = False):
        """Delete task and related output files"""
        with self.conn_handler as (conn, cursor):
            if check:
                cursor.execute('SELECT * FROM task_records WHERE task_id = ?', (task_id,))
                task = cursor.fetchone()
                if task is not None and task['status'] not in ['error' , 'killed']:
                    if not force:
                        raise ValueError(f"Task ID {task_id} is not an error task , cannot be deleted")
                    else:
                        Logger.warning(f"Task ID {task_id} is not an error task , but force is True , so it will be deleted")
            cursor.execute("DELETE FROM task_exit_files WHERE task_id = ?", (task_id,))
            cursor.execute("DELETE FROM task_records WHERE task_id = ?", (task_id,))
            if cursor.rowcount == 0:
                Logger.alert1(f"Task ID {task_id} not found, nothing to delete")
            else:
                Logger.success(f"Task ID {task_id} successfully deleted")

    def del_queue(self, queue_id: str):
        """Delete queue and related tasks"""
        assert False , 'not implemented'
        with self.conn_handler as (conn, cursor):
            cursor.execute("DELETE FROM task_queues WHERE queue_id = ?", (queue_id,))
            cursor.execute("DELETE FROM queue_records WHERE queue_id = ?", (queue_id,))
            if cursor.rowcount == 0:
                Logger.alert1(f"Queue ID {queue_id} not found, nothing to delete")
            else:
                Logger.success(f"Queue ID {queue_id} successfully deleted")
    
    def get_backup_paths(self) -> list[Path]:
        """Return a list of all backup database file paths."""
        return self.conn_handler.all_backup_paths()

    def restore_backup(self , backup_path : Path | str) -> None:
        """Clear the live database and restore from a backup file.

        Parameters
        ----------
        backup_path:
            Path to the SQLite backup file to restore from.
        """
        self.clear_database()
        self.conn_handler.restore(backup_path , delete_backup = True)

class TaskQueue:
    """Ordered, in-memory collection of :class:`TaskItem` objects backed by :class:`TaskDatabase`.

    Maintains at most *max_queue_size* items (oldest evicted first).  Supports
    filtered views, batch refresh from the database, and task status counts.

    Parameters
    ----------
    queue_id:
        Name of the queue to load; defaults to the most recently created queue.
    max_queue_size:
        Maximum number of items to keep in memory (None = unlimited).
    task_db:
        Shared database instance; a new one is created if not supplied.
    """
    _instance : TaskQueue | None = None

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , queue_id : str | None = None , max_queue_size : int | None = 100 , task_db : TaskDatabase | None = None) -> None:
        assert max_queue_size is None or max_queue_size > 0 , 'max_queue_size must be None or greater than 0'
        self.task_db = task_db or TaskDatabase()
        self.queue_id = queue_id or self.task_db.active_queue()
        self.task_db.new_queue(self.queue_id , exist_ok = True)
        self.max_queue_size = max_queue_size
        self.reload()
    
    def __iter__(self):
        """Iterate over task IDs in queue insertion order."""
        return iter(self.queue.keys())

    def __len__(self) -> int:
        """Return the number of tasks currently held in the queue."""
        return len(self.queue)

    def __repr__(self) -> str:
        """Return a debug string showing queue_id, size limit, and current length."""
        return f"TaskQueue(queue_id={self.queue_id},max_queue_size={self.max_queue_size},length={len(self)})"

    def __contains__(self, item : TaskItem) -> bool:
        """Return True if *item* is present in the queue."""
        return item in self.queue.values()

    def get(self, task_id : str | None = None) -> 'TaskItem | None':
        """Return the :class:`TaskItem` with the given *task_id*, or None."""
        if task_id is None:
            return None
        return self.queue.get(task_id)

    def keys(self):
        """Return task-ID keys from the underlying queue dict."""
        return self.queue.keys()

    def values(self):
        """Return :class:`TaskItem` values from the underlying queue dict."""
        return self.queue.values()

    def items(self):
        """Return ``(task_id, TaskItem)`` pairs from the underlying queue dict."""
        return self.queue.items()

    @property
    def empty(self) -> bool:
        """True if the queue contains no task items."""
        return properties.empty(self.queue)

    def reload(self) -> None:
        """Re-fetch all queue tasks from the database and do an initial refresh."""
        self.queue = {task.id: task.set_task_db(self.task_db) for task in self.task_db.get_queue_tasks(self.queue_id, self.max_queue_size)}
        self.refresh()

    def queue_content(self) -> dict[str, dict]:
        """Return a snapshot dict of ``{task_id: task_dict}`` after refreshing."""
        self.refresh()
        return {task.id: task.to_dict() for task in self.queue.values()}

    def add(self, item : TaskItem) -> None:
        """Append *item* to the queue, evicting the oldest entry if over the size limit.

        Also persists the association in the database.
        """
        assert item.id not in self.queue , f'TaskItem {item.id} already exists'
        self.queue[item.id] = item

        self.task_db.del_queue_task(self.queue_id, item.id)
        self.task_db.add_queue_task(self.queue_id, item.id)
        if self.max_queue_size and len(self.queue) > self.max_queue_size:
            self.queue.pop(list(self.queue.keys())[0])

    def create_item(self , script : Path | str | None , source : str | None = None) -> 'TaskItem':
        """Create a new :class:`TaskItem`, persist it, and add it to this queue."""
        item = TaskItem.create(script , self.task_db , source = source)
        self.add(item)
        return item

    def delist(self, item : TaskItem) -> None:
        """Remove *item* from the in-memory queue and its database queue association.

        The underlying task record is NOT deleted.
        """
        if item.id in self.queue:
            self.queue.pop(item.id)
        self.task_db.del_queue_task(self.queue_id, item.id)

    def remove(self, item : TaskItem , force : bool = False) -> None:
        """Remove *item* from the queue and permanently delete its task record.

        Parameters
        ----------
        force:
            If True, delete even tasks not in an error/killed state.
        """
        if item.id in self.queue:
            self.queue.pop(item.id)
        self.task_db.del_queue_task(self.queue_id, item.id)
        self.task_db.del_task(item.id , check = True , force = force)

    def clear_queue_only(self) -> None:
        """Remove all items from the in-memory queue and its database associations without deleting task records."""
        for key in list(self.queue.keys()):
            self.queue.pop(key)
        self.task_db.clear_queue(self.queue_id)

    def clear(self) -> None:
        """Remove all items from the queue and wipe the entire database (with backup)."""
        for key in list(self.queue.keys()):
            self.queue.pop(key)
        self.task_db.clear_database()

    def count(self, status : Literal['starting', 'running', 'complete', 'error' , 'killed']) -> int:
        """Return the number of tasks with the given *status*."""
        return [item.status for item in self.queue.values()].count(status)

    def refresh(self , backend_only : bool = False) -> dict[str, dict[str, Any]]:
        """Pull updates for running and backend-updated tasks from the database.

        Returns True (or a non-empty list) if any task changed status, False (or
        an empty list) otherwise.
        """
        item_ids = self.task_db.get_backend_updated_tasks()
        if not backend_only:
            item_ids += [item.id for item in self.queue.values() if item.is_running]
        if not item_ids:
            return {}
        item_ids = list(set(item_ids))
        changed = {item_id: self.queue[item_id].refresh() for item_id in item_ids if item_id in self.queue}
        if orphan_ids := [item_id for item_id in item_ids if item_id not in self.queue]:
            self.task_db.clear_backend_updated_tasks(orphan_ids)
        return {k : v for k, v in changed.items() if v}
    
    def sync(self) -> None:
        '''sync tasks in record into current queue'''
        self.task_db.sync_queue(self.queue_id)
        self.reload()
            
    def status_message(self , queue : dict[str, 'TaskItem'] | None = None) -> str:
        """Return a pipe-separated summary string of task status counts.

        Example: ``"Total: 5 | Running: 1 | Complete: 3 | Error: 1"``
        """
        if queue is None:
            queue = self.queue
        status = [item.status for item in queue.values()]
        counts = {
            'total': len(status),
            'running': status.count('running'),
            'complete': status.count('complete'),
            'error': status.count('error')
        }
        msg = ' | '.join([f"{k.title()}: {v}" for k, v in counts.items()])
        return msg

    def source_message(self , queue : dict[str, 'TaskItem'] | None = None) -> str:
        """Return a pipe-separated summary string of task source counts (py/app/bash/other)."""
        if queue is None:
            queue = self.queue
        source = [item.source for item in queue.values()]
        counts = {
            'total': len(source),
            'py': source.count('py'),
            'app': source.count('app'),
            'bash': source.count('bash')
        }
        counts['other'] = 2 * len(source) - sum(counts.values())
        msg = ' | '.join([f"{k.title()}: {v}" for k, v in counts.items()])
        return msg

    def filter(self, status : str | None = None,
               source : str | None = None,
               folder : list[Path] | None = None,
               file : list[Path] | None = None ,
               queue : dict[str, 'TaskItem'] | None = None) -> dict[str, 'TaskItem']:
        """Return a filtered, sorted copy of the queue.

        Parameters
        ----------
        status:
            Filter by status string (e.g. ``'running'``); ``'all'`` or None skips.
        source:
            Filter by source (``'py'``, ``'app'``, ``'bash'``, ``'other'``).
        folder:
            Keep only tasks whose script resides inside one of these folders.
        file:
            Keep only tasks whose script path is in this exact list.
        queue:
            Alternative queue dict to filter; defaults to ``self.queue``.
        """
        if queue is None: 
            queue = self.queue.copy()
        else:
            queue = {k: v for k, v in queue.items()}
        if status and status.lower() != 'all':
            queue = {k: v for k, v in queue.items() if v.status == status.lower()}
        if source:
            if source.lower() == 'all':
                ...
            elif source.lower() == 'other':
                queue = {k: v for k, v in queue.items() if v.source not in ['py', 'bash', 'app']}
            else:
                queue = {k: v for k, v in queue.items() if v.source == source.lower()}
        if folder:
            queue = {k: v for k, v in queue.items() if any(v.path.is_relative_to(f) for f in folder)}
        if file:
            queue = {k: v for k, v in queue.items() if v.path in file}
        return {item.id: item for item in self.sort(queue)}   
    
    def latest_n(self , num : int = 10 , script_key : str | None = None) -> dict[str, 'TaskItem']:
        """Return the *num* most-recently created tasks, optionally filtered by *script_key*."""
        if script_key is None:
            d = self.queue.copy()
        else:
            d = {k:v for k,v in self.queue.items() if v.script_key == script_key}
        return {item.id: item for item in self.sort(d)[:num]}
    
    def latest(self , script_key : str | None = None) -> 'TaskItem | None':
        """Return the single most-recently created task, or None if the queue is empty."""
        d = self.latest_n(1 , script_key)
        if d:
            return list(d.values())[0]
        else:
            return None
    
    @classmethod
    def sort(cls , task_items : dict[str, 'TaskItem'] | list['TaskItem'] | Sequence['TaskItem'], key : str = 'create_time' , reverse : bool = True) -> list['TaskItem']:
        """Sort *task_items* by the given attribute *key*, newest first by default."""
        if isinstance(task_items, dict):
            task_items = list(task_items.values())
        return sorted(task_items, key=lambda x: getattr(x, key), reverse=reverse)
    
@dataclass
class TaskItem:
    """Dataclass representing a single pipeline script execution.

    Attributes
    ----------
    script:
        Absolute path to the script file as a string (no spaces, no ``@``).
    cmd:
        The full command string used to launch the script.
    create_time:
        POSIX timestamp when the task was created.
    status:
        Current lifecycle state: ``'starting'``, ``'running'``, ``'complete'``,
        ``'error'``, or ``'killed'``.
    source:
        How the task was launched: ``'py'`` (direct Python), ``'app'``
        (Streamlit UI), or ``'bash'`` (shell).
    pid:
        OS process ID once the subprocess has started.
    start_time, end_time:
        POSIX timestamps for process start and end.
    exit_code:
        Numeric exit code returned by :class:`BackendTaskRecorder`.
    exit_message:
        Human-readable summary from :class:`BackendTaskRecorder`.
    exit_files:
        List of output file paths returned by the script.
    exit_error:
        Traceback or error description on failure.
    task_id:
        Auto-computed on ``__post_init__``; equals ``id``.
    """
    script : str
    cmd : str = ''
    create_time : float = field(default_factory=timestamp)
    status : Literal['starting', 'running', 'complete', 'error' , 'killed'] = 'starting'
    source : Literal['py', 'bash','app'] | str | Any = None
    pid : int | None = None
    start_time : float | None = None
    end_time : float | None = None
    exit_code : int | None = None
    exit_message : str | None = None
    exit_files : list[str] | None = None
    exit_error : str | None = None

    task_id : str | None = None
    queue: TaskQueue | None = None
    
    def __post_init__(self) -> None:
        """Validate fields, normalise ``source``, and initialise transient attributes."""
        assert isinstance(self.script, str) , f'script must be a string, but got {type(self.script)}'
        if self.script:
            assert ' ' not in self.script , f'script must not contain space, but got {self.script}'
            assert '@' not in self.script , f'script must not contain @, but got {self.script}'
        self.source = self.source.lower() if self.source else 'py'
        if self.task_id is None:
            self.task_id = self.id
        else:
            assert self.id.endswith(self.task_id) , f'task_id must be the same as id, but got {self.task_id} and {self.id}'
        self._task_db = None
        self.script_cmd = None
        self._updates_to_sync : dict[str, Any] = {}

    def __eq__(self, other : object) -> bool:
        """Compare by ``id``; also accepts a plain string."""
        if isinstance(other, TaskItem):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        return False
    
    def set_task_db(self , task_db : TaskDatabase | None = None) -> 'TaskItem':
        """Attach a :class:`TaskDatabase` to this item (creates one if not provided). Returns self."""
        if task_db is not None:
            self._task_db = task_db
        elif self._task_db is None:
            self._task_db = TaskDatabase()
        return self
    
    def set_script_cmd(self , script : Path , params : dict | None = None , mode: Literal['shell', 'os'] = 'shell' , **kwargs) -> 'TaskItem':
        """Build and attach the :class:`ScriptCmd`, persisting the resulting ``cmd`` string. Returns self."""
        self.script_cmd = ScriptCmd(script, params, mode, **kwargs)
        self.update({'cmd': str(self.script_cmd)} , sync = True)
        return self

    @property
    def task_db(self) -> TaskDatabase:
        """The attached :class:`TaskDatabase`; raises if not yet set."""
        assert self._task_db is not None , 'task_db is not set'
        return self._task_db

    @property
    def path(self) -> Path:
        """The script path as a :class:`~pathlib.Path` object."""
        return Path(self.script)

    @property
    def relative(self) -> Path:
        """Script path relative to the project scripts root (``PATH.scpt``)."""
        return self.absolute.relative_to(PATH.scpt)

    @property
    def absolute(self) -> Path:
        """Absolute path, resolving cross-machine path differences via the ``scripts`` segment."""
        abs_path = self.path.absolute()
        if abs_path.is_relative_to(PATH.scpt):
            return abs_path
        else:
            return PATH.scpt.joinpath(str(abs_path).split('scripts' , 1)[-1].replace('\\', '/').removeprefix('/'))

    @property
    def stem(self) -> str:
        """Human-readable script name: underscores replaced with spaces, Title Case."""
        return self.path.stem.replace('_', ' ').title()

    @property
    def time_id(self) -> int:
        """Integer POSIX timestamp used as the unique time component of ``id``."""
        return int(self.create_time)

    @property
    def id(self) -> str:
        """Unique task identifier: ``'<relative_script>@<time_id>'``, or ``''`` for anonymous tasks."""
        return f"{str(self.relative)}@{self.time_id}" if self.script else ''

    @property
    def format_path(self) -> str:
        """Human-readable breadcrumb path, e.g. ``'Data > Train Data'``."""
        return ' > '.join(re.sub(r'^\d+ ', '', p).title()
                          for p in Path(self.script_key.replace('_', ' ')).with_suffix('').parts)

    def belong_to(self , script_runner : Any) -> bool:
        """Return True if this task was launched from *script_runner*'s script."""
        return self.script == str(script_runner.script)

    def time_str(self , time_type : Literal['create', 'start', 'end'] = 'create' , format : str = '%Y-%m-%d %H:%M:%S') -> str:
        """Return a formatted datetime string for the given time type.

        Returns ``'N/A'`` on conversion failure.
        """
        try:
            if time_type == 'create':
                return datetime.fromtimestamp(self.create_time).strftime(format)
            elif time_type == 'start':
                return datetime.fromtimestamp(self.start_time or self.create_time).strftime(format)
            elif time_type == 'end':
                return datetime.fromtimestamp(self.end_time or self.create_time).strftime(format)
        except (ValueError , TypeError):
            Logger.error(f'{time_type} time is not a number: {self.create_time}')
            return 'N/A'

    @property
    def script_key(self) -> str:
        """Unique string key matching :attr:`ScriptRunner.script_key`."""
        return str(self.relative)

    @property
    def page_url(self) -> str:
        """Streamlit page URL for this task's script detail page."""
        return runs_page_url(self.script_key)
    
    @classmethod
    def create(cls, script : Path | str | None , task_db : TaskDatabase | None = None , source : Literal['py', 'bash','app'] | str | None = None ,
               queue : TaskQueue | bool | None = None) -> 'TaskItem':
        """Factory: create and persist a new :class:`TaskItem`.

        Parameters
        ----------
        script:
            Path to the script file.  If None, the running ``__main__`` file is
            used (useful when called from inside a backend script).
        task_db:
            Shared database; a new one is created if omitted.
        source:
            Launch origin (``'py'``, ``'bash'``, or ``'app'``).
        queue:
            A :class:`TaskQueue` to register the item with, True to use the
            active queue, or None to skip queue registration.
        """
        if script is None:
            try:
                script = sys.modules['__main__'].__file__
                assert script , 'script is not found'
                script = os.path.abspath(script)
                cmd = ' '.join(sys.argv)
                item = cls(str(script), cmd = cmd , status = 'running' , start_time = timestamp() , source = source)
            except AttributeError as e:
                Logger.error(f'script is not found: {e}')
                return cls('', source = source)
        else:
            item = cls(str(script) , source = source)
        item.set_task_db(task_db)
        item.dump()
        if isinstance(queue, TaskQueue):
            queue.add(item)
            item.queue = queue
        elif queue:
            item.task_db.add_queue_task(item.task_db.active_queue() , item.id)
        return item
    
    @classmethod
    def preview_cmd(cls , script : Path | str | None ,
                    source : Literal['py', 'bash','app'] | str | None = None ,
                    mode: Literal['shell', 'os'] = 'shell' ,
                    **kwargs) -> str:
        """Return the command string that would run *script* without actually executing it."""
        item = cls(str(script) , source = source)
        params = kwargs | {'task_id': item.id , 'source': item.source}
        cmd = ScriptCmd(item.script, params, mode)
        return str(cmd)
    
    @classmethod
    def load(cls , task_id: str , task_db : TaskDatabase | None = None) -> 'TaskItem':
        """Load a persisted :class:`TaskItem` from the database by *task_id*."""
        task_db = task_db or TaskDatabase()
        item = task_db.get_task(task_id)
        assert item is not None , f'Task {task_id} not found'
        item.set_task_db(task_db)
        return item
    
    def refresh(self) -> dict[str, Any]:
        '''refresh task item status , return True if status changed'''
        changed = self.reload()

        if self.pid and self.is_running:
            status = check_process_status(self.pid)
            if status not in ['running', 'complete' , 'disk-sleep' , 'sleeping' , 'zombie']:
                raise ValueError(f"Process {self.pid} is {status}")
            if status in ['complete' , 'zombie']:
                if self.check_killed():
                    changed['status'] = 'killed'
                else:
                    self.update({'status': 'complete'} , sync = True)
                    changed['status'] = 'complete'
                
            elif status != 'running':
                ... # do nothing
                
        if changed and 'status' in changed:
            new_status = changed['status']
            Logger.only_once(f"Task {self.id} status [{new_status.upper()}]" , object = self , mark = f'status_{new_status}' , printer = Logger.info)
            if new_status == 'error' and 'exit_error' in changed:
                Logger.stdout(f"exit_error : {changed['exit_error']}" , color = 'lightred' , indent = 1)
            if 'exit_files' in changed and changed['exit_files']:
                Logger.stdout(f"exit_files : {changed["exit_files"]}" , indent = 1)
        return changed

    def check_killed(self) -> bool:
        """Check for crash-protector files indicating the process was killed.

        If found, updates the task status to ``'killed'``, records exit metadata,
        sends an email notification, and returns True; otherwise returns False.
        """
        crash_protector_paths = self.get_crash_protector()
        if crash_protector_paths:
            updates = {
                'status': 'killed',
                'end_time': timestamp(),
                'exit_code': 1,
                'exit_error': f'CRITICAL: Process {self.pid} is killed , please check the crash_protector files',
                'exit_files': crash_protector_paths,
            }
            self.update(updates , sync = True)
            title = f'Process Killed Unexpectedly'
            body = f"""Process {self.id} killed , information includes:
            - Task ID: {self.id}
            - Script: {self.script}
            - CMD: {self.cmd}
            - PID: {self.pid}
            - Create Time: {self.time_str('create')}
            - Start Time: {self.time_str('start')}
            - End Time: {self.time_str('end')}
            - Duration: {self.duration_str}
            - Exit Code: {self.exit_code}
            - Exit Error: {self.exit_error}
            - Exit Files: {crash_protector_paths}
            """
            Logger.alert3(f'Process Killed Unexpectedly for task {self.id}')
            Email.send(title , body , attachments = crash_protector_paths)
            return True
        else:
            return False

    def wait_until_running(self , * , refresh_interval : int = 1 , refresh_all_interval : int = 5 , starting_timeout : int = 20):
        """wait for running"""
        assert refresh_all_interval % refresh_interval == 0 , f'refresh_all_interval must be a multiple of refresh_interval'
        assert refresh_all_interval > 0 and refresh_interval > 0 , f'refresh_all_interval and refresh_interval must be greater than 0'
        if not self.is_starting:
            return True
        refresh_time = 0
        while self.is_starting:
            if refresh_time >= starting_timeout:
                Logger.error(f'Script {self.script} running timeout! Still starting')
                self.update({
                    'status': 'error' , 'end_time': datetime.now().timestamp() ,
                    'exit_code': 1 ,
                    'exit_error': f'Script {self.script} running timeout! Still starting'} , sync = True)
                return False
            self.refresh()
            if refresh_time % refresh_all_interval == 0:
                if self.queue is not None:
                    self.queue.refresh(backend_only = True)
                elif TaskQueue._instance is not None:
                    TaskQueue._instance.refresh(backend_only = True)
            time.sleep(refresh_interval)
            refresh_time += refresh_interval
        return True

    def wait_until_completion(self , refresh_interval : int = 1 , refresh_all_interval : int = 5 , starting_timeout : int = 20):
        """wait for complete"""
        if not self.is_running:
            return True
        refresh_time = 0
        while self.is_running:
            self.refresh()
            if refresh_time % refresh_all_interval == 0:
                if self.queue is not None:
                    self.queue.refresh(backend_only = True)
                elif TaskQueue._instance is not None:
                    TaskQueue._instance.refresh(backend_only = True)
            time.sleep(refresh_interval)
            refresh_time += refresh_interval
        return True

    def get_crash_protector(self) -> list[str]:
        """Return paths of crash-protector marker files matching this task's ID."""
        if self.task_id is None:
            return []
        long_suffix = '.' + self.task_id.replace('/', '_') + '.'
        crashed_paths = [str(path) for path in PATH.runtime.joinpath('crash_protector').glob('*.md')]
        return [path for path in crashed_paths if long_suffix in path]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialise this item to a dict, omitting None fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def dump(self) -> None:
        """Persist (or overwrite) this item in the attached :class:`TaskDatabase`."""
        self.task_db.new_task(self , overwrite=True)
    
    def reload(self) -> dict[str, Any]:
        """reload task item status from database, return changed items"""
        if not self.task_db.is_backend_updated(self.id):
            return {}

        new_task = self.task_db.get_task(self.id)
        assert new_task is not None , f'Task {self.id} not found'
        changed = {k : v for k, v in new_task.to_dict().items() if v != getattr(self, k) and v is not None}
        if changed:
            self.update(changed)
            if 'status' in changed:
                changed = {'status' : changed['status']} | {k: v for k, v in changed.items() if k != 'status'}
            for k, v in changed.items():
                if k.endswith('_time'):
                    assert v is None or isinstance(v, float) , f'{k} must be a float, but got {type(v)}'
                    changed[k] = f'{v} ({datetime.fromtimestamp(v).strftime('%Y-%m-%d %H:%M:%S')})' if v else 'None'
        self.task_db.del_backend_updated_task(self.id)
        return changed

    def sync(self) -> None:
        """Flush any pending in-memory updates to the database."""
        if self._updates_to_sync:
            self.task_db.update_task(self.id , **self._updates_to_sync)
            self._updates_to_sync = {}
        
    def update(self, updates : dict[str, Any] | None = None , sync : bool = False) -> None:
        """Apply *updates* to this item's attributes and optionally persist to the database.

        Parameters
        ----------
        updates:
            Dict of attribute-name → new-value pairs.
        sync:
            If True, immediately flush to the database via :meth:`sync`.
        """
        if updates is None: 
            return
        [setattr(self, k, v) for k, v in updates.items()]
        self._updates_to_sync = self._updates_to_sync | updates
        if sync:
            self.sync()

    def kill(self) -> bool:
        """Send a kill signal to the process if it is running.

        Returns True on success or if the process is not running, False on failure.
        """
        if self.pid and self.is_running:
            if kill_process(self.pid):
                self.check_killed()
                return True
            else:
                return False
        return True

    @classmethod
    def status_icon(cls , status : Literal['running', 'starting', 'complete', 'error' , 'killed'] , tag : bool = False) -> str:
        """Return a Material icon string for *status*, optionally wrapped in a colour badge."""
        if status in ['running', 'starting']:
            icon , color = ':material/arrow_forward_ios:' , 'blue'
        elif status == 'complete': 
            icon , color = ':material/check:' , 'green'
        elif status == 'error': 
            icon , color = ':material/close:' , 'red'
        elif status == 'killed':
            icon , color = ':material/cancel:' , 'violet'
        else: 
            raise ValueError(f"Invalid status: {status}")
        return f":{color}-badge[{icon}]" if tag else icon
    
    @property
    def status_state(self) -> Literal['running', 'complete', 'error']:
        if self.is_running: 
            return 'running'
        elif self.is_complete:  
            return 'complete'
        elif self.is_error or self.is_killed: 
            return 'error'
        else: 
            raise ValueError(f"Invalid status: {self.status}")

    @property
    def status_title(self) -> str:
        """Human-readable title for the current status (e.g. ``'Running'``, ``'Complete'``)."""
        if self.is_running:
            return 'Running'
        elif self.is_complete:
            return 'Complete'
        elif self.is_error:
            return 'Error'
        elif self.is_killed:
            return 'Killed'
        else:
            raise ValueError(f"Invalid status: {self.status}")

    @property
    def status_color(self) -> str:
        """Streamlit badge colour name for the current status."""
        if self.is_running:
            return 'blue'
        elif self.is_complete:
            return 'green'
        elif self.is_error:
            return 'red'
        elif self.is_killed:
            return 'violet'
        else:
            raise ValueError(f"Invalid status: {self.status}")

    @property
    def is_complete(self) -> bool:
        """True when status is ``'complete'``."""
        return self.status == 'complete'

    @property
    def is_error(self) -> bool:
        """True when status is ``'error'``."""
        return self.status == 'error'

    @property
    def is_killed(self) -> bool:
        """True when status is ``'killed'``."""
        return self.status == 'killed'

    @property
    def is_running(self) -> bool:
        """True when status is ``'running'`` or ``'starting'``."""
        return self.status in ['running', 'starting']

    @property
    def is_starting(self) -> bool:
        """True when status is ``'starting'``."""
        return self.status == 'starting'

    @property
    def plain_icon(self) -> str:
        """Unicode emoji icon for the current status (suitable for plain-text output)."""
        if self.is_running:
            icon = '🔵'
        elif self.is_complete:
            icon = '✅'
        elif self.is_error:
            icon = '❌'
        elif self.is_killed:
            icon = '💀'
        else:
            raise ValueError(f"Invalid status: {self.status}")
        return icon

    @property
    def icon(self) -> str:
        """Material icon string for the current status (without badge)."""
        return self.status_icon(self.status)

    @property
    def tag_icon(self) -> str:
        """Material icon string wrapped in a colour badge for the current status."""
        return self.status_icon(self.status , tag = True)

    @property
    def duration(self) -> float:
        """Elapsed seconds between start (or create) time and end (or now)."""
        start_time = self.start_time or self.create_time
        end_time = self.end_time or timestamp()
        try:
            return end_time - start_time
        except (ValueError , TypeError):
            Logger.error(f'duration is not a number: {end_time} - {start_time}')
            return 0

    @property
    def duration_str(self) -> str:
        """Human-readable duration string (e.g. ``'1m 23s'``)."""
        return Duration(self.duration).fmtstr if self.duration > 0 else 'N/A'

    @property
    def running_str(self) -> str:
        """Streamlit-flavoured markdown summary line for display while the task is running."""
        return f"Script ***{self.format_path} @{self.time_id}*** :gray-badge[Create {self.time_str()}] :orange-badge[Source {self.source.title()}] :violet-badge[PID {self.pid}]"

    def button_str_short(self) -> str:
        """Short label used for task selector buttons (path + HH:MM:SS)."""
        return f"{self.format_path} ({self.time_str(format = '%H:%M:%S')})"

    def button_str_long(self , index : int | None = None , plain_text : bool = False) -> str:
        """Full label used for task history expanders, with status badge and source tag.

        Parameters
        ----------
        index:
            Optional 1-based index prepended to the label.
        plain_text:
            If True, return ASCII-only text (no Streamlit markdown).
        """
        if plain_text:
            if index is not None:
                s = [f"{index}." , self.plain_icon, "."]
            else:
                s = [self.plain_icon, "."]
            s += [f"{self.format_path}" , 
                  f"--Create {self.time_str()}" ,
                  f"--Status {self.status.title()}" ,
                  f"--Source {self.source.title()}"]
        else:
            s = [f"{self.tag_icon}"]
            if index is not None: 
                s += [f"{index: <2}."]
            s.append(f"{self.format_path}")
            s.append(f":gray-badge[Create {self.time_str()}]")
            s.append(f":{self.status_color}-badge[Status {self.status.title()}]")
            s.append(f":orange-badge[Source {self.source.title()}]")
        return " ".join(s)
    
    def button_help_text(self) -> str:
        """Return a single-line tooltip string with full task metadata."""
        return ' | '.join([f"ID: {self.id}" , 
                           f"PID: {self.pid}" , 
                           f"Beg: {self.time_str('start')}" , 
                           f"End: {self.time_str('end')}" , 
                           f"Dur: {self.duration_str}" ,
                           f"Exit Code: {self.exit_code}"])

    def info_list(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all' ,
                  sep_exit_files : bool = True) -> list[tuple[str, str]]:
        """Return a list of ``(label, value)`` pairs describing this task.

        Parameters
        ----------
        info_type:
            Which group of fields to include: ``'enter'`` (ID, script, PID,
            timing), ``'exit'`` (exit code/message/files), or ``'all'`` (both).
        sep_exit_files:
            If True, each exit file gets its own row; otherwise they are joined.
        """
        self.refresh()
        enter_info : list[tuple[str, str]] = []
        if info_type in ['all' , 'enter']:
            enter_info.append(('Item ID', self.id))
            enter_info.append(('Source', self.source))
            enter_info.append(('Script Path', str(self.absolute)))
            enter_info.append(('PID', str(self.pid) if self.pid else 'N/A'))
            enter_info.append(('Create Time', self.time_str('create')))
            enter_info.append(('Start Time', self.time_str('start')))
            enter_info.append(('End Time', self.time_str('end')))
            enter_info.append(('Duration', self.duration_str))
            enter_info.append(('Status', self.status))
        exit_info : list[tuple[str, str]] = []
        if info_type in ['all' , 'exit']:
            if self.exit_code is not None:
                exit_info.append(('Exit Code', f'{self.exit_code}'))
            if self.exit_error is not None:
                exit_info.append(('Exit Error', f'{self.exit_error}'))
            if self.exit_message:
                exit_info.append(('Exit Message', f'{self.exit_message}'))
            if self.exit_files:
                if sep_exit_files:
                    for i , file in enumerate(self.exit_files):
                        path = Path(file).absolute()
                        if path.is_relative_to(PATH.scpt):
                            path = path.relative_to(PATH.scpt)
                        exit_info.append((f'Exit File ({i})', f'{path}'))
                else:
                    exit_info.append(('Exit Files', '\n'.join(self.exit_files)))
        return enter_info + exit_info
        
    def dataframe(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all') -> 'pd.DataFrame':
        """Return task metadata as a two-column DataFrame (Item / Value)."""
        data_list = self.info_list(info_type = info_type)
        df = pd.DataFrame(data_list , columns = pd.Index(['Item', 'Value']))
        return df
    
    def run_script(self , as_workspace: str | None = None , from_workspace: str | None = None) -> 'TaskItem':
        """Launch the script subprocess, capture the PID, and update status to ``'running'``.

        Parameters
        ----------
        as_workspace, from_workspace:
            Forwarded to :meth:`ScriptCmd.run` for workspace-switching support.

        Returns self.
        """
        assert self.script_cmd is not None , 'script cmd is not set'
        try:
            self.script_cmd.run(as_workspace=as_workspace, from_workspace=from_workspace)
        except Exception as e:
            self.update({'status': 'error', 'exit_error': str(e), 'end_time': timestamp()} , sync = True)
            Logger.print_exc(e)
            raise
        return self