import os , sys , re , time
import pandas as pd
from typing import Any , Literal , Sequence
from dataclasses import dataclass , field , asdict
from datetime import datetime
from pathlib import Path

from src.proj import PATH , Logger , Duration
from src.proj.func import check_process_status , kill_process
from src.proj.util import ScriptCmd , DBConnHandler , Email

def timestamp():
    return datetime.now().timestamp()
class TaskDatabase:
    def __init__(self , db_name: str | Path | None = None):
        self.db_path = self.get_db_path()
        if db_name is not None:
            self.db_path = self.db_path.with_name(f'{db_name}.db')
        self.db_name = self.db_path.stem
        self.conn_handler = DBConnHandler(self.db_path)
        self.initialize_database()

    @staticmethod
    def get_db_path():
        return PATH.app_db / 'task_manager.db'

    def initialize_database(self):
        """Initialize database and tables"""
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
            
            # Create indexes for script and status columns
            
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
                    task_id TEXT PRIMARY KEY,
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
            
    def is_empty(self):
        """Check if database is empty"""
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

    def task_count(self):
        """Get task count"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT COUNT(*) FROM task_records')
            return cursor.fetchone()[0]
            
    def clear_database(self):
        """Clear database , will backup the database before clearing"""
        self.conn_handler.backup()
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_records')
            cursor.execute('DELETE FROM task_exit_files')
            cursor.execute('DELETE FROM queue_records')
            cursor.execute('DELETE FROM task_queues')
            cursor.execute('DELETE FROM task_backend_updated')
            cursor.execute('DELETE FROM sqlite_sequence')
        self.initialize_database()
    
    def new_task(self, task: 'TaskItem' , overwrite: bool = False):
        """Insert new task record"""
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

    def new_queue(self, queue_id: str , exist_ok: bool = False):
        """Insert new queue record"""
        with self.conn_handler as (conn, cursor):
            if exist_ok:
                cursor.execute('SELECT * FROM queue_records WHERE queue_id = ?', (queue_id,))
                if cursor.fetchone() is not None: 
                    return
            cursor.execute('''
            INSERT INTO queue_records (queue_id, create_time)
                VALUES (?, ?)
                ''', (queue_id, timestamp()))
    
    def update_task(self, task_id: str, backend_updated: bool = False, **kwargs):
        """Update task status and related information"""
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
                cursor.execute('''
                INSERT INTO task_backend_updated (task_id, updated_time)
                VALUES (?, ?)
                ''', (task_id, timestamp()))

        new_status = self.check_task_status(task_id)
        if 'status' in kwargs:
            if kwargs['status'] != new_status:
                raise ValueError(f"Task {task_id} status update failed , expected {kwargs['status']} but got {new_status}")
            else:
                Logger.success(f"Task {task_id} status updated : {",".join([f"{k}={v}" for k, v in kwargs.items()])}")

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
        
    def clear_backend_updated_tasks(self):
        """Clear backend updated tasks"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_backend_updated')

    def is_backend_updated(self, task_id: str) -> bool:
        """Check if task is backend updated"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_backend_updated WHERE task_id = ?', (task_id,))
            return cursor.fetchone() is not None
        
    def del_backend_updated_task(self, task_id: str):
        """Delete backend updated task"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_backend_updated WHERE task_id = ?', (task_id,))

    def sync_queue(self, queue_id: str):
        """Sync historical tasks into current queue"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT task_id FROM task_records')
            task_ids = [row['task_id'] for row in cursor.fetchall()]
            cursor.execute('DELETE FROM task_queues WHERE queue_id = ?', (queue_id,))
            for task_id in task_ids:
                cursor.execute('INSERT INTO task_queues (queue_id, task_id) VALUES (?, ?)', (queue_id, task_id))

    def clear_queue(self, queue_id: str):
        """Update queue status and related information"""
        with self.conn_handler as (conn, cursor):
            cursor.execute("DELETE FROM task_queues WHERE queue_id = ?", (queue_id,))
    
    def get_task(self, task_id: str) -> 'TaskItem | None':
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
    
    def get_tasks(self, task_ids: list[str]) -> list['TaskItem']:
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
    
    def add_queue_task(self, queue_id: str , task_id: str):
        with self.conn_handler as (conn, cursor):
            cursor.execute('INSERT INTO task_queues (queue_id, task_id) VALUES (?, ?)', (queue_id, task_id))
    
    def del_queue_task(self, queue_id: str , task_id: str):
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_queues WHERE queue_id = ? AND task_id = ?', (queue_id, task_id))

    def get_queue_tasks(self, queue_id: str , max_queue_size: int | None = None) -> list['TaskItem']:
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
    
    def get_backup_paths(self):
        return self.conn_handler.all_backup_paths()
    
    def restore_backup(self , backup_path : Path | str):
        self.clear_database()
        self.conn_handler.restore(backup_path , delete_backup = True)

class TaskQueue:
    def __init__(self , queue_id : str | None = None , max_queue_size : int | None = 100 , task_db : TaskDatabase | None = None):
        assert max_queue_size is None or max_queue_size > 0 , 'max_queue_size must be None or greater than 0'
        self.task_db = task_db or TaskDatabase()
        self.queue_id = queue_id or self.task_db.active_queue()
        self.task_db.new_queue(self.queue_id , exist_ok = True)
        self.max_queue_size = max_queue_size
        self.reload()
    
    def __iter__(self):
        return iter(self.queue.keys())
    
    def __len__(self):
        return len(self.queue)
    
    def __repr__(self):
        return f"TaskQueue(queue_id={self.queue_id},max_queue_size={self.max_queue_size},length={len(self)})"
    
    def __contains__(self, item : 'TaskItem'):
        return item in self.queue.values()
    
    def get(self, task_id : str | None = None):
        if task_id is None: 
            return None
        return self.queue.get(task_id)

    def keys(self):
        return self.queue.keys()
    
    def values(self):
        return self.queue.values()
    
    def items(self):
        return self.queue.items()
    
    def is_empty(self):
        return not self.queue

    def reload(self):
        self.queue = {task.id: task.set_task_db(self.task_db) for task in self.task_db.get_queue_tasks(self.queue_id, self.max_queue_size)}
        self.refresh()

    def queue_content(self):
        self.refresh()
        return {task.id: task.to_dict() for task in self.queue.values()}

    def add(self, item : 'TaskItem'):
        assert item.id not in self.queue , f'TaskItem {item.id} already exists'
        self.queue[item.id] = item
        
        self.task_db.del_queue_task(self.queue_id, item.id)
        self.task_db.add_queue_task(self.queue_id, item.id)
        if self.max_queue_size and len(self.queue) > self.max_queue_size:
            self.queue.pop(list(self.queue.keys())[0])

    def create_item(self , script : Path | str | None , source : str | None = None):
        item = TaskItem.create(script , self.task_db , source = source)
        self.add(item)
        return item
    
    def delist(self, item : 'TaskItem'):
        if item.id in self.queue:
            self.queue.pop(item.id)
        self.task_db.del_queue_task(self.queue_id, item.id)
        
    def remove(self, item : 'TaskItem' , force : bool = False):
        if item.id in self.queue:
            self.queue.pop(item.id)
        self.task_db.del_queue_task(self.queue_id, item.id)
        self.task_db.del_task(item.id , check = True , force = force)

    def empty(self):
        for key in list(self.queue.keys()):
            self.queue.pop(key)
        self.task_db.clear_queue(self.queue_id)

    def clear(self):
        for key in list(self.queue.keys()):
            self.queue.pop(key)
        self.task_db.clear_database()

    def count(self, status : Literal['starting', 'running', 'complete', 'error' , 'killed']):
        return [item.status for item in self.queue.values()].count(status)
    
    def refresh(self):
        backend_updated_item_ids = self.task_db.get_backend_updated_tasks()
        running_item_ids = [item.id for item in self.queue.values() if item.is_running]
        item_ids = list(set(backend_updated_item_ids + running_item_ids))
        changed = [self.queue[item_id].refresh() for item_id in item_ids if item_id in self.queue]
        return changed and any(changed)
    
    def sync(self):
        '''sync tasks in record into current queue'''
        self.task_db.sync_queue(self.queue_id)
        self.reload()
            
    def status_message(self , queue : dict[str, 'TaskItem'] | None = None):
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
    
    def source_message(self , queue : dict[str, 'TaskItem'] | None = None):
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
               queue : dict[str, 'TaskItem'] | None = None):
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
    
    def latest_n(self , num : int = 10 , script_key : str | None = None):
        if script_key is None:
            d = self.queue.copy()
        else:
            d = {k:v for k,v in self.queue.items() if v.script_key == script_key}
        return {item.id: item for item in self.sort(d)[:num]}
    
    def latest(self , script_key : str | None = None):
        d = self.latest_n(1 , script_key)
        if d:
            return list(d.values())[0]
        else:
            return None
    
    @classmethod
    def sort(cls , task_items : dict[str, 'TaskItem'] | list['TaskItem'] | Sequence['TaskItem'], key : str = 'create_time' , reverse : bool = True):
        if isinstance(task_items, dict):
            task_items = list(task_items.values())
        return sorted(task_items, key=lambda x: getattr(x, key), reverse=reverse)
    
@dataclass
class TaskItem:
    '''TaskItem is a class that represents a task item in the Task Queue'''
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
    
    def __post_init__(self):
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
        self._script_cmd = None

    def __eq__(self, other):
        if isinstance(other, TaskItem):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        return False
    
    def set_task_db(self , task_db : TaskDatabase | None = None):
        if task_db is not None:
            self._task_db = task_db
        elif self._task_db is None:
            self._task_db = TaskDatabase()
        return self
    
    def set_script_cmd(self , script_cmd : ScriptCmd):
        self._script_cmd = script_cmd
        self.update({'cmd': str(script_cmd)} , write_to_db = True)
        return self

    @property
    def task_db(self):
        assert self._task_db is not None , 'task_db is not set'
        return self._task_db

    @property
    def path(self):
        return Path(self.script)

    @property
    def relative(self):
        return self.absolute.relative_to(PATH.scpt)
    
    @property
    def absolute(self):
        abs_path = self.path.absolute()
        if abs_path.is_relative_to(PATH.scpt):
            return abs_path
        else:
            return PATH.scpt.joinpath(str(abs_path).split('scripts' , 1)[-1].replace('\\', '/').removeprefix('/'))

    @property
    def stem(self):
        return self.path.stem.replace('_', ' ').title()
    
    @property
    def time_id(self):
        return int(self.create_time)

    @property
    def id(self):
        return f"{str(self.relative)}@{self.time_id}" if self.script else ''
    
    @property
    def format_path(self):
        return ' > '.join(re.sub(r'^\d+ ', '', p).title() 
                          for p in Path(self.script_key.replace('_', ' ')).with_suffix('').parts)

    def belong_to(self , script_runner):
        return self.script == str(script_runner.script)
    
    def time_str(self , time_type : Literal['create', 'start', 'end'] = 'create' , format : str = '%Y-%m-%d %H:%M:%S'):
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
    def script_key(self):
        return str(self.relative)
    
    @classmethod
    def create(cls, script : Path | str | None , task_db : TaskDatabase | None = None , source : Literal['py', 'bash','app'] | str | None = None , 
               queue : TaskQueue | bool | None = None):
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
        elif queue:
            item.task_db.add_queue_task(item.task_db.active_queue() , item.id)
        return item
    
    @classmethod
    def preview_cmd(cls , script : Path | str | None , 
                    source : Literal['py', 'bash','app'] | str | None = None , 
                    mode: Literal['shell', 'os'] = 'shell' , 
                    **kwargs):
        item = cls(str(script) , source = source)
        params = kwargs | {'task_id': item.id , 'source': item.source}
        cmd = ScriptCmd(item.script, params, mode)
        return str(cmd)
    
    @classmethod
    def load(cls , task_id: str , task_db : TaskDatabase | None = None):
        task_db = task_db or TaskDatabase()
        item = task_db.get_task(task_id)
        assert item is not None , f'Task {task_id} not found'
        item.set_task_db(task_db)
        return item
    
    def refresh(self) -> bool:
        '''refresh task item status , return True if status changed'''
        changed = False
        if self.task_db.is_backend_updated(self.id):
            self.reload()
            self.task_db.del_backend_updated_task(self.id)  
            changed = True

        if self.pid and self.is_running:
            status = check_process_status(self.pid)
            if status not in ['running', 'complete' , 'disk-sleep' , 'sleeping' , 'zombie']:
                raise ValueError(f"Process {self.pid} is {status}")
            if status in ['complete' , 'zombie']:
                if not self.check_killed():
                    self.update({'status': 'complete'} , write_to_db = True)
                    return True
            elif status != 'running':
                self.update({'status': 'running'} , write_to_db = False)
                return True
        return changed

    def check_killed(self) -> bool:
        crash_protector_paths = self.get_crash_protector()
        if crash_protector_paths:
            updates = {
                'status': 'killed',
                'end_time': timestamp(),
                'exit_code': 1,
                'exit_error': f'CRITICAL: Process {self.pid} is killed , please check the crash_protector files',
                'exit_files': crash_protector_paths,
            }
            self.update(updates , write_to_db = True)
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

    def wait_until_completion(self , starting_timeout : int = 20):
        """wait for complete"""
        if not self.is_running:
            return True
        while self.is_running:
            self.refresh()
            if self.status == 'starting':
                starting_timeout = starting_timeout - 1
            if starting_timeout <= 0:
                Logger.error(f'Script {self.script} running timeout! Still starting')
                self.update({
                    'status': 'error' , 'end_time': datetime.now().timestamp() ,
                    'exit_code': 1 ,
                    'exit_error': f'Script {self.script} running timeout! Still starting'} , write_to_db = True)
                return False
            time.sleep(1)
        return True

    def get_crash_protector(self) -> list[str]:
        if self.task_id is None:
            return []
        long_suffix = '.' + self.task_id.replace('/', '_') + '.'
        crashed_paths = [str(path) for path in PATH.runtime.joinpath('crash_protector').glob('*.md')]
        return [path for path in crashed_paths if long_suffix in path]
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def dump(self):
        self.task_db.new_task(self , overwrite=True)
    
    def reload(self):
        new_task = self.task_db.get_task(self.id)
        assert new_task is not None , f'Task {self.id} not found'
        changed_items = {k : v for k, v in new_task.to_dict().items() if v != getattr(self, k)}
        if changed_items:
            Logger.success(f"Task {self.id} status updated : {",".join([f"{k}={v}" for k, v in changed_items.items() if v is not None])}")
            self.update(changed_items)
        return self
    
    def update(self, updates : dict[str, Any] | None = None , write_to_db : bool = False):
        if updates is None: 
            return
        [setattr(self, k, v) for k, v in updates.items()]
        if write_to_db:
            self.task_db.update_task(self.id , **updates)

    def kill(self):
        if self.pid and self.is_running:
            if kill_process(self.pid):
                self.check_killed()
                return True
            else:
                return False
        return True

    @classmethod
    def status_icon(cls , status : Literal['running', 'starting', 'complete', 'error' , 'killed'] , tag : bool = False):
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
    def status_title(self):
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
    def status_color(self):
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
    def is_complete(self):
        return self.status == 'complete'

    @property
    def is_error(self):
        return self.status == 'error'

    @property
    def is_killed(self):
        return self.status == 'killed'
    
    @property
    def is_running(self):
        return self.status in ['running', 'starting']

    @property
    def plain_icon(self):
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
    def icon(self):
        return self.status_icon(self.status)
    
    @property
    def tag_icon(self):
        return self.status_icon(self.status , tag = True)

    @property
    def duration(self):
        start_time = self.start_time or self.create_time
        end_time = self.end_time or timestamp()
        try:
            return end_time - start_time
        except (ValueError , TypeError):
            Logger.error(f'duration is not a number: {end_time} - {start_time}')
            return 0
    
    @property
    def duration_str(self):
        return Duration(self.duration).fmtstr
        
    @property
    def running_str(self):
        return f"Script ***{self.format_path} @{self.time_id}*** :gray-badge[Create {self.time_str()}] :orange-badge[Source {self.source.title()}] :violet-badge[PID {self.pid}]"
    
    def button_str_short(self):
        return f"{self.format_path} ({self.time_str(format = '%H:%M:%S')})"
    
    def button_str_long(self , index : int | None = None , plain_text : bool = False):
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
    
    def button_help_text(self):
        return ' | '.join([f"ID: {self.id}" , 
                           f"PID: {self.pid}" , 
                           f"Beg: {self.time_str('start')}" , 
                           f"End: {self.time_str('end')}" , 
                           f"Dur: {self.duration_str}" ,
                           f"Exit Code: {self.exit_code}"])

    def info_list(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all' ,
                  sep_exit_files : bool = True) -> list[tuple[str, str]]:
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
        
    def dataframe(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all'):
        data_list = self.info_list(info_type = info_type)
        df = pd.DataFrame(data_list , columns = pd.Index(['Item', 'Value']))
        return df
    
    def run_script(self):
        cmd = self._script_cmd
        assert cmd is not None , 'script cmd is not set'
        try:
            start_time = timestamp()
            process = cmd.run()
            self.update({'pid': process.real_pid, 'status': 'running', 'start_time': start_time} , write_to_db = True)
        except Exception as e:
            self.update({'status': 'error', 'exit_error': str(e), 'end_time': timestamp()} , write_to_db = True)
            Logger.print_exc(e)
            raise e
        return self

if __name__ == '__main__':
    db = TaskDatabase()

    task = TaskItem(
        script = "/scripts/0_check/0_test_streamlit.py",
        cmd = "/scripts/0_check/0_test_streamlit.py --email 1 --port_name a --module_name bbb --short_test None --forget True --start None --end None --seed 42.0 --task_id 0_check/0_test_streamlit.py@1752898666; exit\"' ",
        create_time = 1752898666.1410549,
        status = "error",
        pid = 10210,
        source = "test",
        start_time = 1752898666.737761,
        end_time = 1752898671.069558,
        exit_code = 1,
        exit_message = "INFO : info:Bye, World!\nERROR : error:Bye, World!\nWARNING : warning:Bye, World!\nDEBUG : debug:Bye, World!\nCRITICAL : critical:Bye, World!",
        exit_files = ["logs/autorun/message_catcher/test_streamlit.20250719001751.html"],
        exit_error = "error:Bye, World!"
    )

    db.new_task(task)

    # 更新任务为运行中
    db.update_task(
        task_id="0_check/0_test_streamlit.py@1752898666",
        status="running",
        pid=12345,
        start_time=timestamp()
    )

    # 更新任务为已完成
    db.update_task(
        task_id="0_check/0_test_streamlit.py@1752898666",
        status="completed",
        end_time=timestamp(),
        exit_code=0
    )



