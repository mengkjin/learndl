import sqlite3 , time , os , sys , json , re , traceback , shutil
from typing import Any , Literal , Sequence , ClassVar
from dataclasses import dataclass , field , asdict
from datetime import datetime
from pathlib import Path
import pandas as pd

from src_runs.util.db import BASE_DIR , runs_db_path , runs_db_dir , exit_message_file
from src_runs.util.abc import check_process_status , kill_process
from src_runs.util.db import queue_json_file , exit_message_file , exit_msg_dir

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
        if db_name is None:
            self.db_path = runs_db_path
        else:
            self.db_path = runs_db_dir / f'{db_name}.db'
        self.db_name = self.db_path.stem
        self.conn_handler = DBConnHandler(self.db_path)
        self.initialize_database(self.db_path)

    @staticmethod
    def initialize_database(db_path: str | Path):
        """Initialize database and tables"""
        # create 4 main tables : 
        # task_records : task records
        # task_exit_files : task exit files
        # queue_records : queue records , last one is the active queue
        # task_queues : task queues and their task_ids
            
        with DBConnHandler(db_path)(check_same_thread = True) as (conn, cursor):
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
                    exit_error TEXT,
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
                CREATE TABLE IF NOT EXISTS queue_records (
                    queue_id TEXT PRIMARY KEY,
                    create_time REAL NOT NULL,
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
    
    def new_task(self, task: 'TaskItem'):
        """Insert new task record"""
        with self.conn_handler as (conn, cursor):
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

    def new_queue(self, queue_id: str):
        """Insert new queue record"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('''
            INSERT INTO queue_records (queue_id, create_time)
            VALUES (?, ?)
            ''', (queue_id, time.time()))
    
    def update_task(self, task_id: str, **kwargs):
        """Update task status and related information"""
        if not kwargs: return
        with self.conn_handler as (conn, cursor):
            query = ' '.join([
                "UPDATE task_records SET",
                ", ".join([f"{k} = ?" for k in kwargs.keys()]) + ",",
                "WHERE task_id = ?"
            ])
            params = list(kwargs.values()) + [task_id]
            cursor.execute(query, params)
            if exit_files := kwargs.pop('exit_files' , None):
                cursor.execute("DELETE FROM task_exit_files WHERE task_id = ?", (task_id,))
                for file_path in exit_files:
                    cursor.execute('''
                    INSERT INTO task_exit_files (task_id, file_path)
                    VALUES (?, ?)
                    ''', (task_id, file_path))

    def update_queue(self, queue_id: str, task_ids: list[str]):
        """Update queue status and related information"""
        with self.conn_handler as (conn, cursor):
            cursor.execute("DELETE FROM task_queues WHERE queue_id = ?", (queue_id,))
            for task_id in task_ids:
                cursor.execute('''
                INSERT INTO task_queues (queue_id, task_id)
                VALUES (?, ?)
                ''', (queue_id, task_id))
    
    def get_task(self, task_id: str) -> 'TaskItem | None':
        """Get task information"""
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_records WHERE task_id = ?', (task_id,))
            task = cursor.fetchone()
            if not task: return None

            cursor.execute('SELECT file_path FROM task_exit_files WHERE task_id = ?', (task_id,))
            files = [row['file_path'] for row in cursor.fetchall()]

            item = TaskItem.load(dict(task) | {'exit_files': files})
        return item
    
    def get_tasks(self, task_ids: list[str]) -> list['TaskItem']:
        """Get tasks information"""
        tasks = []
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT * FROM task_records WHERE task_id IN (?)', (task_ids,))
            tasks = [TaskItem.load(dict(row)) for row in cursor.fetchall()]
        return tasks
    
    def get_queue(self, queue_id: str) -> 'TaskQueue':
        """Get queue information"""
        tasks = []
        with self.conn_handler as (conn, cursor):
            cursor.execute('SELECT task_id FROM task_queues WHERE queue_id = ?', (queue_id,))
            tasks = self.get_tasks([row['task_id'] for row in cursor.fetchall()])
            queue = TaskQueue(queue_id)
            queue.load(tasks)
        return queue
    
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
    _instances : ClassVar[dict[str, 'TaskQueue']] = {}
    
    def __init__(self , queue_name : str = 'default' , max_queue_size : int | None = 100):
        assert max_queue_size is None or max_queue_size > 0 , 'max_queue_size must be None or greater than 0'
        self.queue_name = queue_name
        self.queue_file = queue_json_file(queue_name)
        self.max_queue_size = max_queue_size
        self.queue : dict[str, 'TaskItem'] = {}
        self.load()
        self._instances[self.queue_name] = self
    
    def __iter__(self):
        return iter(self.queue.keys())
    
    def __len__(self):
        return len(self.queue)
    
    def __repr__(self):
        return f"TaskQueue(queue_name={self.queue_name},max_queue_size={self.max_queue_size},length={len(self)})"
    
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

    def load(self , tasks: list['TaskItem'] | None = None):
        if tasks is not None:
            self.queue = {task.id: task for task in tasks}
            self.save()
            return
        content = self.full_queue_dict()
        task_ids = list(content.keys())
        if self.max_queue_size:
            task_ids = task_ids[-self.max_queue_size:]
        for item_id in task_ids:
            try:
                if item_id in self.queue:
                    self.queue[item_id].update(**content.get(item_id , {}))
                else:
                    self.add(TaskItem.load(content[item_id]))
            except Exception as e:
                raise ValueError(f'Error loading task {item_id}: {e}')
        
    def queue_content(self):
        if self.queue_file.exists():
            with open(self.queue_file, 'r') as f:
                content = f.read() or "{}"
        else:
            content = "{}"
        return content
    
    def queue_dict(self) -> dict[str, Any]:
        return json.loads(self.queue_content())
    
    def merge_exit_message(self):
        exit_message = self.exit_message_dict()
        for task_id in exit_message:
            try:
                if task_id in self.queue:
                    self.queue[task_id].update(**exit_message[task_id])
                else:
                    self.queue[task_id] = TaskItem.load(exit_message[task_id])
            except Exception as e:
                raise ValueError(f'Error merging exit message for task {task_id}: {e}')
        self.save()

    @classmethod
    def exit_message_dict(cls , task_ids : Sequence | dict | None = None , delete_after_load : bool = True) -> dict[str, Any]:
        exit_message = {}
        if task_ids is None:
            files = list(exit_msg_dir.glob('*.json'))
        else:
            files = [exit_message_file(task_id) for task_id in task_ids]
        for file in files:
            if not file.exists(): continue
            with open(file, 'r') as f:
                exit_message.update(json.load(f))
            if delete_after_load: file.unlink()
        return exit_message
    
    def full_queue_dict(self):
        queue : dict[str , Any] = json.loads(self.queue_content())
        task_ids = list(queue.keys())
        exit_message : dict[str, Any] = self.exit_message_dict(task_ids)
        full_queue = {k : queue.get(k) | exit_message.get(k , {}) for k in task_ids}
        full_queue.update({k : v for k, v in exit_message.items() if k not in task_ids})
        self.save(full_queue)
        return full_queue

    def save(self , content : dict[str, Any] | None = None):
        if content is None:
            content = {k: v.to_dict() for k, v in self.queue.items()}
        with open(self.queue_file, 'w') as f:
            json.dump(content, f, indent=2)

    def add(self, item : 'TaskItem'):
        assert item.id not in self.queue , f'TaskItem {item.id} already exists'
        self.queue[item.id] = item
        if self.max_queue_size and len(self.queue) > self.max_queue_size:
            self.queue.pop(list(self.queue.keys())[0])
        self.save()

    def create_item(self, script : Path | str):
        item = TaskItem.create(script , self)
        return item

    def remove(self, item : 'TaskItem'):
        if item.id in self.queue:
            self.queue.pop(item.id)
            self.save()

    def clear(self):
        for key in list(self.queue.keys()):
            if self.queue[key].status != 'running': self.queue.pop(key)
        self.save()

    def count(self, status : Literal['starting', 'running', 'complete', 'error']):
        return [item.status for item in self.queue.values()].count(status)
    
    def refresh(self):
        status_changed = False
        for item in self.queue.values():
            changed = item.refresh()
            if changed: status_changed = True
        if status_changed: self.save()

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
    
    def __post_init__(self):
        assert isinstance(self.script, str) , f'script must be a string, but got {type(self.script)}'
        assert ' ' not in self.script , f'script must not contain space, but got {self.script}'
        assert '@' not in self.script , f'script must not contain @, but got {self.script}'
        
    def __eq__(self, other):
        if isinstance(other, TaskItem):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        return False

    @property
    def path(self):
        return Path(self.script)

    @property
    def relative(self):
        return self.absolute.relative_to(BASE_DIR)
    
    @property
    def absolute(self):
        abs_path = self.path.absolute()
        if abs_path.is_relative_to(BASE_DIR):
            return abs_path
        else:
            return BASE_DIR.joinpath(str(abs_path).split('src_runs' , 1)[-1].removeprefix('/'))

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
    def load(cls, item : dict[str, Any]):
        return cls(**{k: v for k, v in item.items() if k != 'task_id'})
    
    @classmethod
    def create(cls, script : Path | str , queue : Any = None):
        item = cls(str(script))
        if queue is not None: queue.add(item)
        return item
    
    @classmethod
    def create_from_sys(cls, queue : Any = None):
        script = sys.modules['__main__'].__file__
        assert script , 'script is not found'
        script = os.path.abspath(script)
        item = cls(str(script))
        item.update(cmd = ' '.join(sys.argv) , pid = os.getpid() , status = 'running' , start_time = time.time())
        if queue is not None: queue.add(item)
        return item
    
    def refresh(self):
        '''refresh task item status , return True if status changed'''
        changed = False
        if self.pid and self.status not in ['complete', 'error']:
            status = check_process_status(self.pid)
            if status not in ['running', 'complete' , 'disk-sleep' , 'sleeping']:
                raise ValueError(f"Process {self.pid} is {status}")
            if status in ['complete' , 'disk-sleep' , 'sleeping']:
                self.status = 'complete'
                self.end_time = time.time()
                changed = True
        if self.load_exit_message():
            changed = True
        return changed
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def update(self, **updates):
        if updates:
            [setattr(self, k, v) for k, v in updates.items()]

    def kill(self):
        if self.pid and self.status == 'running':
            if kill_process(self.pid):
                self.update(status = 'complete', end_time = time.time())
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

    def load_exit_message(self):
        file = exit_message_file(self.id)
        if not file.exists(): return False
        with open(file, 'r') as f:
            exit_message = json.load(f)
        self.update(**exit_message.get(self.id , {}))
        return True
    
    def info_list(self , info_type : Literal['all' , 'enter' , 'exit'] = 'all'):
        self.load_exit_message()
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
                    if path.is_relative_to(BASE_DIR):
                        path = path.relative_to(BASE_DIR)
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



