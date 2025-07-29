import sqlite3 , time , os , sys , re , shutil
from typing import Any , Literal
from dataclasses import dataclass , field , asdict
from datetime import datetime
from pathlib import Path
import pandas as pd

from src_ui.db import RUNS_DIR , get_task_db_path
from src_ui.abc import check_process_status , kill_process

class DBConnHandler:
    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        self.reset()

    def reset(self):
        self.check_same_thread = False
        
    @staticmethod
    def get_connection(db_path: str | Path , check_same_thread: bool = True):
        """Get database connection(using Streamlit cache)"""
        conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row  # allow to access rows as dictionaries
        return conn
    
    def __call__(self , check_same_thread = False):
        self.check_same_thread = check_same_thread
        return self
        
    def __enter__(self):
        self.conn = self.get_connection(self.db_path , check_same_thread = bool(self.check_same_thread))
        self.conn.__enter__()
        self.cursor = self.conn.cursor()
        return self.conn , self.cursor
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.reset()
        self.conn.__exit__(exc_type, exc_value, exc_tb)

class TaskDatabase:
    def __init__(self , db_name: str | Path | None = None):
        self.db_path = get_task_db_path()
        if db_name is not None:
            self.db_path = self.db_path.with_name(f'{db_name}.db')
        self.db_name = self.db_path.stem
        self.conn_handler = DBConnHandler(self.db_path)
        self.initialize_database()

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
            
    def clear_database(self):
        """Clear database , will backup the database before clearing"""
        self.backup()
        with self.conn_handler as (conn, cursor):
            cursor.execute('DELETE FROM task_records')
            cursor.execute('DELETE FROM task_exit_files')
            cursor.execute('DELETE FROM queue_records')
            cursor.execute('DELETE FROM task_queues')
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
                if cursor.fetchone() is not None: return
            cursor.execute('''
            INSERT INTO queue_records (queue_id, create_time)
                VALUES (?, ?)
                ''', (queue_id, time.time()))
    
    def update_task(self, task_id: str, backend_updated: bool = False, **kwargs):
        """Update task status and related information"""
        if not kwargs: return
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
                ''', (task_id, time.time()))

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
            if not task: return None

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
            tasks = [TaskItem(**dict(row)) for row in cursor.fetchall()]
        return tasks
    
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
            if max_queue_size: task_ids = task_ids[-max_queue_size:]
            return self.get_tasks(task_ids)
    
    def active_queue(self) -> str:
        '''get the last queue in table queue_records'''
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT queue_id FROM queue_records ORDER BY create_time DESC LIMIT 1')
            return cursor.fetchone()['queue_id']
    
    def del_task(self, task_id: str , verbose: bool = False):
        """Delete task and related output files"""
        assert False , 'not implemented'
        with self.conn_handler as (conn, cursor):
            cursor.execute("DELETE FROM task_exit_files WHERE task_id = ?", (task_id,))
            cursor.execute("DELETE FROM task_records WHERE task_id = ?", (task_id,))
            if verbose:
                if cursor.rowcount == 0:
                    print(f"Task ID {task_id} not found, nothing to delete")
                else:
                    print(f"Task ID {task_id} successfully deleted")

    def del_queue(self, queue_id: str , verbose: bool = False):
        """Delete queue and related tasks"""
        assert False , 'not implemented'
        with self.conn_handler as (conn, cursor):
            cursor.execute("DELETE FROM task_queues WHERE queue_id = ?", (queue_id,))
            cursor.execute("DELETE FROM queue_records WHERE queue_id = ?", (queue_id,))
            if verbose:
                if cursor.rowcount == 0:
                    print(f"Queue ID {queue_id} not found, nothing to delete")
                else:
                    print(f"Queue ID {queue_id} successfully deleted")
    
    def backup(self, suffix: str | None = None):
        """
        Backup database and rename the original database
        :param suffix: backup suffix, if not specified, use timestamp
        :return: new database path
        """
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.db_name}_{suffix}.db"
        backup_path = self.db_path.with_name(backup_name)
        shutil.copy(self.db_path, backup_path)
        return backup_path
    

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
        if task_id is None: return None
        return self.queue.get(task_id)

    def keys(self):
        return self.queue.keys()
    
    def values(self):
        return self.queue.values()
    
    def items(self):
        return self.queue.items()
    
    def empty(self):
        return not self.queue

    def reload(self):
        self.queue = {task.id: task.set_task_db(self.task_db) for task in self.task_db.get_queue_tasks(self.queue_id, self.max_queue_size)}
        for task in self.queue.values():
            task.set_task_db(self.task_db)

    def queue_content(self):
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
        
    def remove(self, item : 'TaskItem'):
        if item.id in self.queue:
            self.queue.pop(item.id)
        self.task_db.del_queue_task(self.queue_id, item.id)

    def clear(self):
        for key in list(self.queue.keys()):
            if self.queue[key].status != 'running': self.queue.pop(key)
        self.task_db.clear_queue(self.queue_id)

    def count(self, status : Literal['starting', 'running', 'complete', 'error']):
        return [item.status for item in self.queue.values()].count(status)
    
    def refresh(self):
        return any(item.refresh() for item in self.queue.values())
    
    def sync(self):
        '''sync tasks in record into current queue'''
        self.task_db.sync_queue(self.queue_id)
        self.reload()
            
    def status_message(self):
        status = [item.status for item in self.queue.values()]
        return f"Running: {status.count('running')} | Complete: {status.count('complete')} | Error: {status.count('error')}"

    def filter(self, status : Literal['all' , 'starting', 'running', 'complete', 'error'] | None = None,
               folder : list[Path] | None = None,
               file : list[Path] | None = None):
        filtered_queue = self.queue.copy()
        if status and status.lower() != 'all':
            filtered_queue = {k: v for k, v in filtered_queue.items() if v.status == status.lower()}
        if folder:
            filtered_queue = {k: v for k, v in filtered_queue.items() if any(v.path.is_relative_to(f) for f in folder)}
        if file:
            filtered_queue = {k: v for k, v in filtered_queue.items() if v.path in file}
        return filtered_queue   
    
    def latest(self , num : int = 10):
        return {item.id: item for item in list(self.queue.values())[-num:]}
    
@dataclass
class TaskItem:
    '''TaskItem is a class that represents a task item in the Task Queue'''
    script : str
    cmd : str = ''
    create_time : float = field(default_factory=time.time)
    status : Literal['starting', 'running', 'complete', 'error'] = 'starting'
    source : str | None = None
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
        assert ' ' not in self.script , f'script must not contain space, but got {self.script}'
        assert '@' not in self.script , f'script must not contain @, but got {self.script}'
        if self.task_id is None:
            self.task_id = self.id
        else:
            assert self.task_id == self.id , f'task_id must be the same as id, but got {self.task_id} and {self.id}'
        self._task_db = None

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

    @property
    def task_db(self):
        assert self._task_db is not None , 'task_db is not set'
        return self._task_db

    @property
    def path(self):
        return Path(self.script)

    @property
    def relative(self):
        return self.absolute.relative_to(RUNS_DIR)
    
    @property
    def absolute(self):
        abs_path = self.path.absolute()
        if abs_path.is_relative_to(RUNS_DIR):
            return abs_path
        else:
            return RUNS_DIR.joinpath(str(abs_path).split('src_runs' , 1)[-1].removeprefix('/'))

    @property
    def stem(self):
        return self.path.stem.replace('_', ' ').title()
    
    @property
    def time_id(self):
        return int(self.create_time)

    @property
    def id(self):
        return f"{str(self.relative)}@{self.time_id}"
    
    @property
    def format_path(self):
        return ' > '.join(re.sub(r'^\d+ ', '', p).title() for p in str(self.relative).removesuffix('.py').replace('_', ' ').split('/'))

    @property
    def button_str(self):
        return f"{self.format_path} ({self.time_str()})"
    
    def belong_to(self , script_runner):
        return self.script == str(script_runner.script)
    
    def time_str(self , time_type : Literal['create', 'start', 'end'] = 'create'):
        if time_type == 'create':
            time = self.create_time
        elif time_type == 'start':
            time = self.start_time
        elif time_type == 'end':
            time = self.end_time
        if time is None: return 'N/A'
        return datetime.fromtimestamp(time).strftime('%H:%M:%S')

    @property
    def runner_script_key(self):
        return str(self.relative)
    
    @classmethod
    def create(cls, script : Path | str | None , task_db : TaskDatabase | None = None , source : str | None = None , 
               queue : TaskQueue | None = None):
        if script is None:
            script = sys.modules['__main__'].__file__
            assert script , 'script is not found'
            script = os.path.abspath(script)
            cmd = ' '.join(sys.argv)
            item = cls(str(script), cmd = cmd , status = 'running' , start_time = time.time() , source = source)
        else:
            item = cls(str(script) , source = source)
        item.set_task_db(task_db)
        item.dump()
        if queue is not None: queue.add(item)
        return item
    
    @classmethod
    def load(cls , task_id: str , task_db : TaskDatabase | None = None):
        task_db = task_db or TaskDatabase()
        item = task_db.get_task(task_id)
        assert item is not None , f'Task {task_id} not found'
        item.set_task_db(task_db)
        return item
    
    def refresh(self):
        '''refresh task item status , return True if status changed'''
        changed = False
        if self.pid and self.status not in ['complete', 'error']:
            status = check_process_status(self.pid)
            if status not in ['running', 'complete' , 'disk-sleep' , 'sleeping']:
                raise ValueError(f"Process {self.pid} is {status}")
            if status in ['complete' , 'disk-sleep' , 'sleeping']:
                self.update({'status': 'complete', 'end_time': time.time()} , write_to_db = True)
                changed = True
        if self.task_db.is_backend_updated(self.id):
            self.reload()
            changed = True
            self.task_db.del_backend_updated_task(self.id)
        return changed
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def dump(self):
        self.task_db.new_task(self , overwrite=True)
    
    def reload(self):
        new_task = self.task_db.get_task(self.id)
        assert new_task is not None , f'Task {self.id} not found'
        self.update(new_task.to_dict())
    
    def update(self, updates : dict[str, Any] | None = None , write_to_db : bool = False):
        if updates is None: return
        [setattr(self, k, v) for k, v in updates.items()]
        if write_to_db:
            self.task_db.update_task(self.id , **updates)

    def kill(self):
        if self.pid and self.status == 'running':
            if kill_process(self.pid):
                self.update({'status': 'complete', 'end_time': time.time()} , write_to_db = True)
                return True
            else:
                return False
        return True

    @classmethod
    def status_icon(cls , status : Literal['running', 'starting', 'complete', 'error'] , tag : bool = False):
        if status in ['running', 'starting']: 
            icon , color = ':material/arrow_forward_ios:' , 'green'
        elif status == 'complete': 
            icon , color = ':material/check:' , 'green'
        elif status == 'error': 
            icon , color = ':material/close:' , 'red'
        else: raise ValueError(f"Invalid status: {status}")
        return f":{color}-badge[{icon}]" if tag else icon
    
    @property
    def status_state(self):
        if self.status in ['running', 'starting']: return 'running'
        elif self.status == 'complete':  return 'complete'
        elif self.status == 'error': return 'error'
        else: raise ValueError(f"Invalid status: {self.status}")

    @property
    def plain_icon(self):
        if self.status in ['running', 'starting']: 
            icon = '🟡'
        elif self.status == 'complete': 
            icon = '✅'
        elif self.status == 'error': 
            icon = '❌'
        else: raise ValueError(f"Invalid status: {self.status}")
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
        end_time = self.end_time or time.time()
        return end_time - start_time
    
    @property
    def duration_str(self):
        duration = self.duration
        if duration < 60:
            return f"{duration:.2f} Secs"
        elif duration < 3600:
            return f"{int(duration / 60)} Min {int(duration%60)} Secs"
        else:
            return f"{int(duration / 3600)} Hr {int(duration%3600 / 60)} Min {int(duration%60)} Secs"
    
    def info_list(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all'):
        self.reload()
        enter_info , exit_info = [] , []
        if info_type in ['all' , 'enter']:
            enter_info.extend([
                ['Item ID', self.id],
                ['Script Path', str(self.absolute)],
                ['PID', str(self.pid) if self.pid else 'N/A'],
                ['Create Time', self.time_str('create')],
                ['Start Time', self.time_str('start')],
                ['End Time', self.time_str('end')], 
                ['Duration', self.duration_str],
                ['Status', self.status],
            ])
        if info_type in ['all' , 'exit']:
            if self.exit_code is not None:
                exit_info.append(['Exit Code', f'{self.exit_code}'])
            if self.exit_error is not None:
                exit_info.append(['Exit Error', f'{self.exit_error}'])
            if self.exit_message:
                exit_info.append(['Exit Message', f'{self.exit_message}'])
            if self.exit_files:
                for i , file in enumerate(self.exit_files):
                    path = Path(file).absolute()
                    if path.is_relative_to(RUNS_DIR):
                        path = path.relative_to(RUNS_DIR)
                    exit_info.append([f'Exit File ({i})', f'{path}'])
        return enter_info + exit_info
        
    def dataframe(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all'):
        data_list = self.info_list(info_type = info_type)
        df = pd.DataFrame(data_list , columns = ['Item', 'Value'])
        return df

if __name__ == '__main__':
    db = TaskDatabase()

    task = TaskItem(
        script = "/src_runs/0_check/0_test_streamlit.py",
        cmd = "/src_runs/0_check/0_test_streamlit.py --email 1 --port_name a --module_name bbb --short_test None --forget True --start None --end None --seed 42.0 --task_id 0_check/0_test_streamlit.py@1752898666; exit\"' ",
        create_time = 1752898666.1410549,
        status = "error",
        pid = 10210,
        source = "test",
        start_time = 1752898666.737761,
        end_time = 1752898671.069558,
        exit_code = 1,
        exit_message = "INFO : info:Bye, World!\nERROR : error:Bye, World!\nWARNING : warning:Bye, World!\nDEBUG : debug:Bye, World!\nCRITICAL : critical:Bye, World!",
        exit_files = ["logs/autorun/message_capturer/test_streamlit.20250719001751.html"],
        exit_error = "error:Bye, World!"
    )

    db.new_task(task)

    # 更新任务为运行中
    db.update_task(
        task_id="0_check/0_test_streamlit.py@1752898666",
        status="running",
        pid=12345,
        start_time=time.time()
    )

    # 更新任务为已完成
    db.update_task(
        task_id="0_check/0_test_streamlit.py@1752898666",
        status="completed",
        end_time=time.time(),
        exit_code=0
    )



