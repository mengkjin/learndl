from pathlib import Path
from typing import Literal , Any , Sequence , ClassVar
import re , yaml , json , time , os , subprocess , shutil
import pandas as pd
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass , asdict , field

from src_runs.util.abc import check_process_status , kill_process , terminal_cmd , get_real_pid
from src_runs.util.st_file import BASE_DIR , queue_json_file , exit_message_file
    
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
    def iter_folder(cls, folder_path: Path | str = BASE_DIR, level: int = 0 , 
                    ignore_starters = ('.', '_') ,
                    ignore_files = ('widget.py', 'streamlit.py' , 'util' , 'st') ,
                    min_level: int = 0 , max_level: int = 2):
        '''get all valid items from folder recursively'''
        items : list[cls] = []
        if level < min_level or level > max_level: return items
        folder_path = Path(folder_path)
        assert folder_path.is_dir() , f'{folder_path} is not a folder'
            
        for item in folder_path.iterdir():
            if item.name.startswith(ignore_starters) or item.name in ignore_files: continue
            if item.is_dir() and not list(item.iterdir()): continue
            items.append(cls(item , level))
            if item.is_dir(): 
                items.extend(cls.iter_folder(item , level + 1 , ignore_starters , ignore_files, min_level , max_level))
        
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

    def run_script(self , queue : 'TaskQueue | None' = None , close_after_run = False , **kwargs) -> 'TaskItem':
        '''run script and return exit code (0: error, 1: success)'''
        item = TaskItem.create(self.script , queue)
        item.cmd = terminal_cmd(self.script, kwargs | {'task_id': item.id}, close_after_run=close_after_run)
        
        try:
            process = subprocess.Popen(item.cmd, shell=True, encoding='utf-8')
            pid = get_real_pid(process , item.cmd)
            item.update(pid = pid, status = 'running', start_time = time.time())
        except Exception as e:
            # update queue status to error
            item.update(status = 'error', error = str(e), end_time = time.time())
            raise e
        finally:
            if queue is not None: queue.save()
        return item

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

    def load(self):
        content = self.full_queue_dict()
        task_ids = list(content.keys())
        if self.max_queue_size:
            task_ids = task_ids[-self.max_queue_size:]
        for item_id in task_ids:
            if item_id in self.queue:
                self.queue[item_id].update(**content.get(item_id , {}))
            else:
                self.add(TaskItem.load(content[item_id]))
        
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
        exit_message = self.exit_message_dict(self.queue , True)
        for task_id in self.queue:
            self.queue[task_id].update(**exit_message.get(task_id , {}))
        self.save()

    def exit_message_dict(self , task_ids : Sequence | dict , delete_after_load : bool = True) -> dict[str, Any]:
        exit_message = {}
        for task_id in task_ids:
            file = exit_message_file(task_id)
            if not file.exists(): continue
            with open(file, 'r') as f:
                exit_message.update(json.load(f))
            if delete_after_load: file.unlink()
        return exit_message
    
    def full_queue_dict(self):
        queue : dict[str , Any] = json.loads(self.queue_content())
        task_ids = list(queue.keys())
        exit_message : dict[str, Any] = self.exit_message_dict(task_ids)
        full_queue = {k : queue[k] | exit_message.get(k , {}) for k in task_ids}
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
        return f"{self.format_path} ({self.time_str()})"
    
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

    @property
    def runner_script_key(self):
        return str(self.relative)

    @classmethod
    def load(cls, item : dict[str, Any]):
        return cls(**item)
    
    @classmethod
    def create(cls, script : Path | str , queue : 'TaskQueue | None' = None):
        item = cls(str(script))
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
    
    @property
    def status_state(self):
        if self.status in ['running', 'starting']: return 'running'
        elif self.status == 'complete':  return 'complete'
        elif self.status == 'error': return 'error'
        else: raise ValueError(f"Invalid status: {self.status}")

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
    