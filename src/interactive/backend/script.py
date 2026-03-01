import re , yaml
from pathlib import Path
from typing import Literal , Any , ClassVar
from dataclasses import dataclass , asdict , field

from src.proj import PATH , Logger , MACHINE  # noqa
from src.proj.util import ScriptCmd , Options # noqa
from .task import TaskItem , TaskQueue
    
@dataclass
class PathItem:
    path: Path
    level: int

    ignore_patterns : ClassVar[tuple[str, ...]] = (r'^db.py$' , r'^util$' , r'^\.(.*)$' , r'^_(.*)$')
    
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
        return self.path.relative_to(PATH.scpt)
    
    @property
    def absolute(self):
        return self.path.absolute()
    
    @classmethod
    def iter_folder(cls, folder_path: Path | str = PATH.scpt, level: int = 0 , 
                    min_level: int = 0 , max_level: int = 2):
        '''get all valid items from folder recursively'''
        items : list['PathItem'] = []
        if level < min_level or level > max_level: 
            return items
        folder_path = Path(folder_path)
        assert folder_path.is_dir() , f'{folder_path} is not a folder'
            
        for item in folder_path.iterdir():
            if any(re.match(pattern, item.name) for pattern in cls.ignore_patterns):
                 continue
            if item.is_dir() and not list(item.iterdir()): 
                continue
            items.append(cls(item , level))
            if item.is_dir(): 
                items.extend(cls.iter_folder(item , level + 1 , min_level , max_level))
        
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
                          for p in Path(self.script_key.replace('_', ' ')).with_suffix('').parts)

    @classmethod
    def from_key(cls , script_key : str , base_dir = PATH.scpt):
        parts = Path(script_key).parts
        return PathItem(base_dir.joinpath(*parts) , len(parts) - 1)

    @classmethod
    def from_path(cls , path : Path):
        if not path.is_relative_to(PATH.scpt):
            raise ValueError(f'{path} is not a relative path to {PATH.scpt} , connot convert to PathItem')
        return cls.from_key(str(path.relative_to(PATH.scpt)))

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

    header_pattern : ClassVar[str] = r'^#(.*)'
    exit_patterns : ClassVar[tuple[str, ...]] = (r'^# exit.*', r'^$')
    ignore_patterns : ClassVar[tuple[str, ...]] = (r'^#!.*', r'^# coding:.*')

    def __post_init__(self):
        if self.mode not in ['shell', 'os']:
            raise ValueError(f'Invalid mode: {self.mode}')

    def get_param_inputs(self):
        return ScriptParamInput.from_dict(self.parameters)

    @classmethod
    def read_from_file(cls , path : Path):
        yaml_lines: list[str] = []
        try:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()
                    if any(re.match(pattern, stripped_line) for pattern in cls.ignore_patterns): 
                        continue
                    elif any(re.match(pattern, stripped_line) for pattern in cls.exit_patterns):
                        break
                    elif match := re.match(cls.header_pattern, stripped_line):
                        yaml_lines.append(match.group(1))
                    
            yaml_str = '\n'.join(line for line in yaml_lines)
            kwargs = yaml.safe_load(yaml_str) or {}
            kwargs['description'] = kwargs.get('description') or path.name
        except FileNotFoundError:
            kwargs = {
                'disabled': True, 
                'content': f'file not found : {path}', 
                'description': 'file not found'
            }
        except yaml.YAMLError as e:
            kwargs = {
                'disabled': True, 
                'content': f'YAML parsing error : {e}', 
                'description': 'YAML parsing error'
            }
        except Exception as e:
            kwargs = {
                'disabled': True, 
                'content': f'read file error : {e}', 
                'description': 'read file error'
            }
            
        return cls(**kwargs)

    @property
    def ready(self):
        """check if the script is ready to run"""
        return not self.disabled and not self.parameters

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
                    Logger.warning(e)
                    Logger.warning(f'Invalid type: {self.type} , using str as default')
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
        
        self.header = ScriptHeader.read_from_file(self.script)

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
        return re.sub(r'^\d+_', '', Path(self.script_key).parts[0]).lower()
    
    @property
    def format_path(self):
        return ' > '.join(re.sub(r'^\d+ ', '', p).title() 
                          for p in Path(self.script_key.replace('_', ' ')).with_suffix('').parts)
    
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
    def ready(self):
        return self.header.ready
    
    @property
    def information(self):
        infos = f'''
{self.content} / 
{self.todo}
'''
        return infos
        
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