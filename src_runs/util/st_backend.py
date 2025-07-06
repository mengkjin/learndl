from pathlib import Path
from typing import Literal , Any
import re , yaml , json , time , os , subprocess , logging
import pandas as pd
from datetime import datetime
from dataclasses import dataclass , asdict , field
import streamlit as st

from .abc import check_process_status , kill_process , terminal_cmd , get_real_pid

BASE_DIR = Path(__file__).parent.parent
assert BASE_DIR.name == 'src_runs' , f'BASE_DIR {BASE_DIR} not right'

backend_dir = BASE_DIR / '_backend'
backend_dir.mkdir(parents=True, exist_ok=True)

log_file = backend_dir / 'st_backend.log'
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"st_backend.py loaded")

@dataclass
class PathItem:
    path: Path
    level: int

    @property
    def name(self):
        return self.path.name
    
    @property
    def is_dir(self):
        return self.path.is_dir()

    @property
    def is_file(self):
        return not self.path.is_dir()
    
    @property
    def relative(self):
        return self.path.relative_to(BASE_DIR)
    
    @property
    def absolute(self):
        return self.path.absolute()
    
    @classmethod
    def iter_folder(cls, folder_path: Path | str = BASE_DIR, level: int = 0 , ignore_starters = ('.', '_' , 'util') ,
                    min_level: int = 0 , max_level: int = 2):
        '''get all valid items from folder recursively'''
        items : list[cls] = []
        if level < min_level or level > max_level: return items
        folder_path = Path(folder_path)
        assert folder_path.is_dir() , f'{folder_path} is not a folder'
            
        for item in folder_path.iterdir():
            if item.name.startswith(ignore_starters): continue
            items.append(cls(item , level))
            if item.is_dir(): 
                items.extend(cls.iter_folder(item , level + 1 , ignore_starters , min_level , max_level))
        
        items.sort(key=lambda x: (x.path))
        return items
    
    def script_runner(self):
        return ScriptRunner(self)

@dataclass
class ScriptHeader:
    coding: str = 'utf-8'
    author: str = 'jinmeng'
    date: str = '2024-11-27'
    description: str = ''
    content: str = ''
    todo: str = ''
    email: bool = False
    close_after_run: bool = False
    param_inputs: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False

    def get_param_inputs(self):
        return ScriptParamInput.from_dict(self.param_inputs)

@dataclass
class ScriptParamInput:
    name: str
    type: Literal['str', 'int', 'float', 'bool', 'list', 'tuple' , 'enum'] | list[str] | tuple[str]
    desc: str
    required: bool = False
    default: Any = None
    min: Any = None
    max: Any = None
    prefix: str = ''
    enum: list[str] | None = None

    @classmethod
    def from_dict(cls, param_inputs: dict[str, dict[str, Any]]):
        return [cls(name = name, **param_inputs[name]) for name in param_inputs]
    
    def as_dict(self):
        return asdict(self)
    
    @property
    def ptype(self):
        if isinstance(self.type, str):
            if self.type == 'str':
                ptype = str
            elif self.type == 'int':
                ptype = int
            elif self.type == 'float':
                ptype = float
            elif self.type == 'bool':
                ptype = bool
            elif self.type in ['list', 'tuple' , 'enum']:
                assert self.enum , f'enum is required for {self.type}'
                ptype = list(self.enum)
            else:
                raise ValueError(f'Invalid type: {self.type}')
        elif isinstance(self.type, (list, tuple)):
            ptype = list(self.type)
        else:
            raise ValueError(f'Invalid type: {self.type}')
        return ptype
    
    @property
    def title(self):
        title = self.name.replace('_', ' ').title()
        return title

    @property
    def placeholder(self):
        placeholder = self.desc if self.desc else self.name
        return placeholder
    
    def is_valid(self , value):
        if self.required:
            return value not in ['', None , 'Choose an option']
        return True
    
    def error_message(self, value):
        if not self.is_valid(value):
            operator = 'input' if self.type in ['str', 'int', 'float'] else 'select'
            return f"Please {operator} a valid value for [{self.title}]"
        return None

class ScriptRunner:
    def __init__(self, path_item: PathItem, base_dir: Path | None = None):
        self.path = path_item
        assert self.script.is_file() and self.script.suffix == '.py', f'{self.script} is not a python script'
        
        self.header = self.parse_header()
        self.base_dir = base_dir

    def __repr__(self):
        return f"ScriptRunner(script={self.script})"

    @property
    def id(self):
        return str(self.script)
    
    def __eq__(self, other):
        if isinstance(other, ScriptRunner):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        return False
        
    @property
    def script(self):
        return self.path.absolute

    @property
    def level(self):
        return self.path.level

    @property
    def script_name(self):
        return re.sub(r'^\d+_', '', self.script.stem).replace('_', ' ').title()

    @property
    def script_key(self):
        return str(self.path.relative)
    
    @property
    def path_parts(self):
        return [re.sub(r'^\d+_', '', part).replace('_', ' ').title() for part in self.path.relative.parts]
    
    @property
    def desc(self):
        return self.header.description.title()
    
    @property
    def disabled(self):
        return self.header.disabled
    
    @property
    def content(self):
        return self.header.content
    
    @property
    def todo(self):
        return self.header.todo
    
    @property
    def information(self):
        infos = f'''
{self.content} / 
{self.todo}
'''
        return infos
        
    def parse_header(self, verbose=False, include_starter='#', exit_starter='', ignore_starters=('#!', '# coding:')) -> ScriptHeader:
        yaml_lines: list[str] = []
        try:
            with open(self.script, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith(ignore_starters): 
                        continue
                    elif stripped_line.startswith(include_starter):
                        yaml_lines.append(stripped_line)
                    elif stripped_line.startswith(exit_starter):
                        break

            yaml_str = '\n'.join(line.removeprefix(include_starter) for line in yaml_lines)
            header = ScriptHeader(**(yaml.safe_load(yaml_str) or {}))
        except FileNotFoundError:
            header = ScriptHeader(
                disabled = True, 
                content = f'file not found : {self.script}', 
                description = 'file not found'
            )
        except yaml.YAMLError as e:
            header = ScriptHeader(
                disabled = True, 
                content = f'YAML parsing error : {e}', 
                description = 'YAML parsing error'
            )
        except Exception as e:
            header = ScriptHeader(
                disabled = True, 
                content = f'read file error : {e}', 
                description = 'read file error'
            )

        if not header.description: header.description = self.script.name
            
        return header

    def run_script(self , close_after_run = False , **kwargs):
        '''run script and return exit code (0: error, 1: success)'''
        item = TaskItem.create(self.script)
        item.cmd = terminal_cmd(self.script, kwargs | {'task_id': item.id}, close_after_run=close_after_run)
        
        try:
            process = subprocess.Popen(item.cmd, shell=True, encoding='utf-8')
            pid = get_real_pid(process , item.cmd)
            item.update(pid = pid, status = 'running', start_time = time.time())
            
        except Exception as e:
            # update queue status to error
            item.update(status = 'error', error = str(e), end_time = time.time())
            raise e
            
        TaskQueue.refresh()
        return item

class TaskQueue:
    '''TaskQueue is a class that represents a queue of tasks'''
    _queue_file = backend_dir / 'task_queue.json'
    _exit_message_file = backend_dir / 'exit_message.json'
    _queue : dict[str, 'TaskItem'] = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.load()
    
    def __iter__(self):
        return iter(self._queue.keys())
    
    @classmethod
    def get(cls, task_id : str | None = None):
        if task_id is None: return None
        return cls._queue.get(task_id)

    @classmethod
    def keys(cls):
        return cls._queue.keys()
    
    @classmethod
    def values(cls):
        return cls._queue.values()
    
    @classmethod
    def items(cls):
        return cls._queue.items()
    
    @classmethod
    def empty(cls):
        return not cls._queue

    @classmethod
    def load(cls):
        content = cls.full_content()
        cls._queue = {key: TaskItem.load(value) for key, value in content.items()}
        cls._exit_message_file.unlink(missing_ok=True)

    @classmethod
    def exit_message_content(cls):
        if cls._exit_message_file.exists():
            with open(cls._exit_message_file, 'r') as f:
                _exit_message = f.read() or "{}"
        else:
            _exit_message = "{}"
        return _exit_message
    
    @classmethod
    def queue_content(cls):
        if cls._queue_file.exists():
            with open(cls._queue_file, 'r') as f:
                _queue_content = f.read() or "{}"
        else:
            _queue_content = "{}"
        return _queue_content
    
    @classmethod
    def full_content(cls):
        _queue : dict[str , Any] = json.loads(cls.queue_content())
        _exit_message : dict[str, Any] = json.loads(cls.exit_message_content())
        if not _exit_message: 
            return _queue
        else:
            _queue = {k: v | _exit_message.get(k , {}) for k, v in _queue.items()}
            cls.save(_queue)
            cls._exit_message_file.unlink()
        return _queue

    @classmethod
    def save(cls , content : dict[str, Any] | None = None):
        if content is None:
            content = {k: v.to_dict() for k, v in cls._queue.items()}
        with open(cls._queue_file, 'w') as f:
            json.dump(content, f , indent=2)

    @classmethod
    def add(cls, item : 'TaskItem'):
        cls.load()
        cls._queue[item.id] = item
        cls.save()

    @classmethod
    def remove(cls, item : 'TaskItem'):
        cls.load()
        if item.id in cls._queue:
            del cls._queue[item.id]
            cls.save()

    @classmethod
    def clear(cls):
        cls.load()
        for key in list(cls._queue.keys()):
            if cls._queue[key].status != 'running': cls._queue.pop(key)
        cls.save()

    @classmethod
    def count(cls, status : Literal['starting', 'running', 'complete', 'error']):
        return [item.status for item in cls._queue.values()].count(status)
    
    @classmethod
    def refresh(cls):
        status_changed = False
        cls.load()
        for item in cls._queue.values():
            changed = item.refresh()
            if changed: status_changed = True
        if status_changed: cls.save()

    @classmethod
    def status_message(cls):
        status = [item.status for item in cls._queue.values()]
        return f"Running: {status.count('running')} | Complete: {status.count('complete')} | Error: {status.count('error')}"

    @classmethod
    def update_exit_message(cls , task_id : str | None , files : list[str] | None = None , code : int | None = None , message : str | None = None):
        if not task_id: return
        if not (files or code or message): return
        if not cls._exit_message_file.exists():
            cls._exit_message_file.touch()
        with open(cls._exit_message_file, 'r+') as f:
            old_exit_message = f.read() or "{}"
            old_exit_message = json.loads(old_exit_message)
            old_exit_message[task_id] = {
                'exit_files': files,
                'exit_code': code,
                'exit_message': message
            }
            f.seek(0)
            f.write(json.dumps(old_exit_message, indent=2))
            f.truncate()
            print(f.read())

@dataclass
class TaskItem:
    '''TaskItem is a class that represents a task item in the Task Queue'''
    script : str
    cmd : str = ''
    create_time : float = field(default_factory=time.time)
    status : Literal['starting', 'running', 'complete', 'error'] = 'starting'
    pid : int | None = None
    start_time : float | None = None
    end_time : float | None = None
    exit_code : int | None = None
    exit_message : str | None = None
    exit_files : list[str] | None = None
    error : str | None = None

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
        return self.path.absolute()

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
        return f":material/terminal: {self.format_path} ({self.time_str()})"
    
    def belong_to(self , runner : ScriptRunner):
        return self.script == str(runner.script)
    
    def time_str(self , time_type : Literal['create', 'start', 'end'] = 'create'):
        if time_type == 'create':
            time = self.create_time
        elif time_type == 'start':
            time = self.start_time
        elif time_type == 'end':
            time = self.end_time
        if time is None: return 'N/A'
        return datetime.fromtimestamp(time).strftime('%H:%M:%S')

    @classmethod
    def load(cls, item : dict[str, Any]):
        return cls(**item)
    
    @classmethod
    def create(cls, script : Path | str):
        item = cls(str(script))
        TaskQueue.add(item)
        return item
    
    def refresh(self):
        if self.pid and self.status != 'complete':
            status = check_process_status(self.pid)
            if status not in ['running', 'complete']:
                raise ValueError(f"Process {self.pid} is {status}")
            if status == 'complete':
                self.status = 'complete'
                self.end_time = time.time()
                return True
        return False
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def update(self, **updates):
        if updates:
            [setattr(self, k, v) for k, v in updates.items()]
            TaskQueue.save()

    def kill(self):
        if self.pid and self.status == 'running':
            if kill_process(self.pid):
                self.update(status = 'complete', end_time = time.time())
                return True
            else:
                return False
        return True

    @property
    def icon(self):
        if self.status in ['running', 'starting']: return ':green-badge[:material/arrow_forward_ios:]'
        elif self.status == 'complete':  return ':green-badge[:material/check:]'
        elif self.status == 'error': return ':red-badge[:material/close:]'
        else: raise ValueError(f"Invalid status: {self.status}")
    

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
    
    @property
    def status_state(self):
        if self.status in ['running', 'starting']: return 'running'
        elif self.status == 'complete':  return 'complete'
        elif self.status == 'error': return 'error'
        else: raise ValueError(f"Invalid status: {self.status}")
    
    def info_list(self , include_exit_info : bool = True):
        data_list = [
            ['Script Path', str(self.relative)],
            ['PID', str(self.pid) if self.pid else 'N/A'],
            ['Create Time', self.time_str('create')],
            ['Start Time', self.time_str('start')],
            ['End Time', self.time_str('end')], 
            ['Duration', self.duration_str],
            ['Status', self.status],
        ]
        if include_exit_info: data_list.extend(self.exit_info_list())
        return data_list
    
    def exit_info_list(self):
        data_list = []
        if self.exit_code is not None:
            data_list.append(['Exit Code', f'{self.exit_code}'])
        if self.exit_message:
            data_list.append(['Exit Message', f'{self.exit_message}'])
        if self.exit_files:
            for i , file in enumerate(self.exit_files):
                path = Path(file).absolute()
                if path.is_relative_to(BASE_DIR):
                    path = path.relative_to(BASE_DIR)
                data_list.append([f'Exit File ({i})', f'{path}'])
        return data_list
        
    def dataframe(self , include_exit_info : bool = True):
        data_list = self.info_list(include_exit_info = include_exit_info)
        df = pd.DataFrame(data_list , columns = ['Item', 'Value'])
        return df

class ExitMessenger:
    '''
    ExitMessenger is a class that manages the exit message of a task
    can pass files , exit_code , message to the task
    '''
        
    @classmethod
    def update(cls , task_id : str | None , files : list[str] | None = None , code : int | None = None , message : str | None = None):
        if not task_id: return
        TaskQueue.update_exit_message(task_id, files, code, message)
        
    