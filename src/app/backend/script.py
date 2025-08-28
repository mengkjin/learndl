from pathlib import Path
from typing import Literal , Any
import re , yaml , time
from dataclasses import dataclass , asdict , field

from src.app.abc import ScriptCmd
from src.app.db import SCPT_DIR

from .task import TaskItem , TaskQueue
    
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
        return self.path.relative_to(SCPT_DIR)
    
    @property
    def absolute(self):
        return self.path.absolute()
    
    @classmethod
    def iter_folder(cls, folder_path: Path | str = SCPT_DIR, level: int = 0 , 
                    ignore_starters = ('.', '_') ,
                    ignore_files = ('db.py' , 'util') ,
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
    
    @property
    def script_key(self):
        return str(self.relative)
    
    @property
    def format_path(self):
        return ' > '.join(re.sub(r'^\d+ ', '', p).title() 
                          for p in self.script_key.removesuffix('.py').replace('_', ' ').replace('\\', '/').split('/'))

    @classmethod
    def from_key(cls , script_key : str , base_dir = SCPT_DIR):
        components = script_key.replace('\\' , '/').split('/')
        return PathItem(base_dir.joinpath(*components) , len(components) - 1)

@dataclass
class ScriptHeader:
    coding: str = 'utf-8'
    author: str = 'jinmeng'
    date: str = '2024-11-27'
    description: str = ''
    content: str = ''
    todo: str = ''
    email: bool = False
    mode: Literal['shell', 'os'] = 'shell'
    parameters: dict[str, Any] = field(default_factory=dict)
    file_editor: dict[str, Any] = field(default_factory=dict)
    file_previewer: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False

    def __post_init__(self):
        if self.mode not in ['shell', 'os']:
            raise ValueError(f'Invalid mode: {self.mode}')

    def get_param_inputs(self):
        return ScriptParamInput.from_dict(self.parameters)

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
                try:
                    ptype = eval(self.type)
                except Exception as e:
                    print(e)
                    print(f'Invalid type: {self.type} , using str as default')
                    ptype = str
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
    def __init__(self, path_item: PathItem):
        self.path = path_item
        assert self.script.is_file() and self.script.suffix == '.py', f'{self.script} is not a python script'
        
        self.header = self.parse_header()

    def __repr__(self):
        return f"ScriptRunner(script={self.script})"
    
    @classmethod
    def from_key(cls , script_key : str):
        return PathItem.from_key(script_key).script_runner()

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
    def script_group(self):
        return re.sub(r'^\d+_', '', self.script_key.replace('\\', '/').split('/')[0]).lower()
    
    @property
    def format_path(self):
        return ' > '.join(re.sub(r'^\d+ ', '', p).title() 
                          for p in self.script_key.removesuffix('.py').replace('_', ' ').replace('\\', '/').split('/'))
    
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

    def build_task(self , queue : TaskQueue | None = None , mode: Literal['shell', 'os'] = 'shell' , **kwargs) -> 'TaskItem':
        '''run script and return exit code (0: error, 1: success)'''

        item = TaskItem.create(self.script , source = 'app' , queue=queue)
        params = kwargs | {'task_id': item.id , 'source': item.source}
        cmd = ScriptCmd(self.script, params, mode)
        item.set_script_cmd(cmd)
        return item
    
    def preview_cmd(self , mode: Literal['shell', 'os'] = 'shell' , **kwargs) -> str:
        '''run script and return exit code (0: error, 1: success)'''
        return TaskItem.preview_cmd(self.script , 'app' , mode , **kwargs)