"""
Script discovery, metadata parsing, and execution abstraction for the interactive app.

Classes
-------
PathItem
    Represents a single file or folder under the scripts root, with helpers for
    recursive discovery and filtering of valid pipeline scripts.
ScriptHeader
    Parses the YAML front-matter embedded in a script's leading comment block to
    expose parameters, readiness status, email flags, and mode settings.
ScriptParamInput
    Type-safe descriptor for a single script parameter, including validation and
    widget-hint metadata used by the frontend.
ScriptRunner
    High-level facade over a :class:`PathItem` that combines header metadata with
    task-building capabilities.
"""
from __future__ import annotations

import re , yaml
from pathlib import Path
from typing import Literal , Any , ClassVar
from dataclasses import dataclass , asdict , field

from src.proj import PATH , Logger , MACHINE  # noqa
from src.proj.util import Options # noqa
from .task import TaskItem , TaskQueue , runs_page_url

def _format_path(script_key : str) -> str:
    """get human-readable breadcrumb path, e.g. ``'Data > Train Data'`` from script_key"""
    return ' > '.join(re.sub(r'^\d+ ', '', p).title()
                      for p in Path(script_key.replace('_', ' ')).with_suffix('').parts)

@dataclass
class PathItem:
    """A single file or directory entry rooted under the scripts folder.

    Attributes
    ----------
    path:
        Absolute :class:`~pathlib.Path` of the item.
    level:
        Nesting depth relative to the scripts root (0 = top-level).
    ignore_patterns:
        Regex patterns that cause an item to be skipped during discovery.
    """
    path: Path
    level: int

    ignore_patterns : ClassVar[tuple[str, ...]] = (r'^db.py$' , r'^util$' , r'^\.(.*)$' , r'^_(.*)$')

    @property
    def name(self) -> str:
        """Basename of the path."""
        return self.path.name

    @property
    def is_dir(self) -> bool:
        """True if this item is a directory."""
        return self.path.is_dir()

    @property
    def is_file(self) -> bool:
        """True if this item is a regular file (not a directory)."""
        return not self.path.is_dir()

    @property
    def relative(self) -> Path:
        """Path relative to the project scripts root (``PATH.scpt``)."""
        return self.path.relative_to(PATH.scpt)

    @property
    def absolute(self) -> Path:
        """Absolute resolved path."""
        return self.path.absolute()

    @classmethod
    def iter_folder(cls, folder_path: Path | str = PATH.scpt, level: int = 0 ,
                    min_level: int = 0 , max_level: int = 2) -> list['PathItem']:
        '''get all valid items from folder recursively'''
        items : list[PathItem] = []
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

    def script_runner(self) -> ScriptRunner:
        """Construct a :class:`ScriptRunner` wrapping this item."""
        return ScriptRunner(self)

    @property
    def script_key(self) -> str:
        """Unique string key: the path relative to the scripts root as a string."""
        return str(self.relative)

    @property
    def format_path(self) -> str:
        """Human-readable breadcrumb path, e.g. ``'Data > Train Data'``."""
        return _format_path(self.script_key)

    @classmethod
    def from_key(cls , script_key : str , base_dir : Path = PATH.scpt) -> PathItem:
        """Construct a :class:`PathItem` from a *script_key* string and optional base directory."""
        parts = Path(script_key).parts
        return PathItem(base_dir.joinpath(*parts) , len(parts) - 1)

    @classmethod
    def from_path(cls , path : Path) -> PathItem:
        """Construct a :class:`PathItem` from an absolute path that must be under ``PATH.scpt``."""
        if not path.is_relative_to(PATH.scpt):
            raise ValueError(f'{path} is not a relative path to {PATH.scpt} , connot convert to PathItem')
        return cls.from_key(str(path.relative_to(PATH.scpt)))

@dataclass
class ScriptHeader:
    """YAML front-matter extracted from a pipeline script's leading comment block.

    The parser reads lines prefixed with ``#`` until it hits a blank line or a
    line matching one of the ``exit_patterns``.  The resulting YAML string is
    loaded into this dataclass.

    Attributes
    ----------
    coding, author, date:
        Informational metadata fields.
    description:
        Short human-readable description shown in the UI.
    content:
        Longer body text (markdown) shown on the script detail page.
    todo:
        Pending work notes shown on the script detail page.
    email:
        Whether to send an email notification on task completion.
    mode:
        Execution mode — ``'shell'`` (via shell subprocess) or ``'os'`` (os.system).
    blacklist:
        Machine-name blocklist; the script is disabled on listed machines.
    parameters:
        Ordered dict of parameter definitions forwarded to :class:`ScriptParamInput`.
    file_editor:
        Config dict for an optional in-page YAML file editor widget.
    file_previewer:
        Config dict for an optional in-page file previewer widget.
    disabled:
        If True the script is treated as unavailable.
    """
    coding: str = 'utf-8'
    author: str = 'jinmeng'
    date: str = '2024-11-27'
    description: str = ''
    content: str = ''
    todo: str = ''
    email: bool = False
    mode: Literal['shell', 'os'] = 'shell'
    blacklist: dict[str, list[str]] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    file_editor: dict[str, Any] = field(default_factory=dict)
    file_previewer: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False

    header_pattern : ClassVar[str] = r'^#(.*)'
    exit_patterns : ClassVar[tuple[str, ...]] = (r'^# exit.*', r'^$')
    ignore_patterns : ClassVar[tuple[str, ...]] = (r'^#!.*', r'^# coding:.*')

    def __post_init__(self) -> None:
        """Validate ``mode`` after dataclass initialisation."""
        if self.mode not in ['shell', 'os']:
            raise ValueError(f'Invalid mode: {self.mode}')

    def get_param_inputs(self) -> list['ScriptParamInput']:
        """Return the list of :class:`ScriptParamInput` objects built from ``parameters``."""
        return ScriptParamInput.from_dict(self.parameters)

    @classmethod
    def read_from_file(cls , path : Path) -> ScriptHeader:
        """Parse YAML front-matter from *path* and return a populated :class:`ScriptHeader`.

        On any read/parse failure a disabled header with an error description is
        returned instead of raising.
        """
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
    def ready(self) -> Literal[0, 1, 2 , 3]:
        """
        check if the script is ready to run
        0: disabled or blacklisted
        1: at least one parameter is required
        2: ready to run with all parameters not required
        3: ready to run with no parameter
        """
        if self.disabled or MACHINE.name in self.blacklist.get('machine', []):
            return 0
        if not self.parameters:
            return 3
        elif not any(param.get('required' , False) for param in self.parameters.values()):
            return 2
        else:
            return 1


@dataclass
class ScriptParamInput:
    """Type-safe descriptor for a single script parameter.

    Attributes
    ----------
    name:
        Parameter name (used as the CLI flag and widget key).
    type:
        Type specifier string (``'str'``, ``'int'``, ``'float'``, ``'bool'``,
        ``'list'``, ``'tuple'``, ``'enum'``) or a list/tuple of string options.
    desc:
        Short description shown as placeholder or tooltip.
    required:
        Whether a non-empty value must be supplied before the script can run.
    default:
        Default value pre-filled in the widget.
    min, max:
        Optional bounds for numeric types.
    prefix:
        CLI prefix prepended to the value when building the command string.
    enum:
        Enumeration values for ``list``/``tuple``/``enum`` types.
    """
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
    def from_dict(cls, param_inputs: dict[str, dict[str, Any]]) -> list['ScriptParamInput']:
        """Build a list of :class:`ScriptParamInput` instances from a raw parameters dict."""
        return [cls(name = name, **param_inputs[name]) for name in param_inputs]

    def as_dict(self) -> dict[str, Any]:
        """Serialise this instance to a plain dict."""
        return asdict(self)

    @property
    def ptype(self) -> type | list[str]:
        """Resolve ``type`` to a Python type object or list of valid string options."""
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
    def title(self) -> str:
        """Human-readable title derived from the parameter name (snake_case → Title Case)."""
        title = self.name.replace('_', ' ').title()
        return title

    @property
    def placeholder(self) -> str:
        """Placeholder text for text/number widgets; falls back to ``name`` when ``desc`` is empty."""
        placeholder = self.desc if self.desc else self.name
        return placeholder

    def is_valid(self , value : Any) -> bool:
        """Return True if *value* satisfies the ``required`` constraint."""
        if self.required:
            return value not in ['', None , 'Choose an option']
        return True

    def error_message(self, value : Any) -> str | None:
        """Return a user-facing error string if *value* is invalid, otherwise None."""
        if not self.is_valid(value):
            operator = 'input' if self.type in ['str', 'int', 'float'] else 'select'
            return f"Please {operator} a valid value for [{self.title}]"
        return None

class ScriptRunner:
    """High-level facade over a :class:`PathItem` that couples header metadata with task building.

    Caches the parsed :class:`ScriptHeader` on construction so subsequent property
    accesses are cheap.

    Parameters
    ----------
    path_item:
        A file-level :class:`PathItem` pointing to a ``.py`` script.
    """
    def __init__(self, path_item: PathItem) -> None:
        """Initialise runner from *path_item*, asserting it is a Python file."""
        self.path = path_item
        assert self.script.is_file() and self.script.suffix == '.py', f'{self.script} is not a python script'

        self.header = ScriptHeader.read_from_file(self.script)

    def __repr__(self) -> str:
        """Return a debug representation showing the script path."""
        return f"ScriptRunner(script={self.script})"

    @classmethod
    def from_key(cls , script_key : str) -> ScriptRunner:
        """Construct a :class:`ScriptRunner` from a *script_key* string."""
        return PathItem.from_key(script_key).script_runner()

    @property
    def id(self) -> str:
        """Unique identifier: the absolute path as a string."""
        return str(self.script)

    def __eq__(self, other : object) -> bool:
        """Compare runners by their ``id``; also accepts a plain string."""
        if isinstance(other, ScriptRunner):
            return self.id == other.id
        elif isinstance(other, str):
            return self.id == other
        return False

    @property
    def script(self) -> Path:
        """Absolute path to the script file."""
        return self.path.absolute

    @property
    def level(self) -> int:
        """Nesting depth of the script relative to the scripts root."""
        return self.path.level

    @property
    def script_name(self) -> str:
        """Display name of the script: numeric prefix stripped, underscores → spaces, Title Case."""
        return re.sub(r'^\d+_', '', self.script.stem).replace('_', ' ').title()

    @property
    def script_key(self) -> str:
        """Unique string key: the path relative to the scripts root."""
        return str(self.path.relative)

    @property
    def page_url(self) -> str:
        """get runs page url"""
        return runs_page_url(self.script_key)

    @property
    def script_group(self) -> str:
        """Top-level group name (e.g. ``'data'``, ``'train'``) with numeric prefix stripped."""
        return re.sub(r'^\d+_', '', Path(self.script_key).parts[0]).lower()

    @property
    def format_path(self) -> str:
        """Human-readable breadcrumb path, e.g. ``'Data > Train Data'``."""
        return _format_path(self.script_key)

    @property
    def path_parts(self) -> list[str]:
        """List of breadcrumb path components with numeric prefixes stripped."""
        return [re.sub(r'^\d+_', '', part).replace('_', ' ').title() for part in self.path.relative.parts]

    @property
    def desc(self) -> str:
        """Short description from the script header (Title Case)."""
        return self.header.description.title()

    @property
    def disabled(self) -> bool:
        """True if the script header marks this script as disabled or blacklisted."""
        return self.header.disabled

    @property
    def content(self) -> str:
        """Long-form content text from the script header."""
        return self.header.content

    @property
    def todo(self) -> str:
        """Pending work notes from the script header."""
        return self.header.todo

    @property
    def ready(self) -> Literal[0, 1, 2, 3]:
        """Readiness status delegated to :attr:`ScriptHeader.ready`."""
        return self.header.ready

    @property
    def information(self) -> str:
        """Combined content and todo text for display."""
        infos = f'''
{self.content} /
{self.todo}
'''
        return infos

    def build_task(self , queue : TaskQueue | None = None , mode: Literal['shell', 'os'] = 'shell' , **kwargs) -> TaskItem:
        """Create and configure a :class:`TaskItem` ready to execute this script.

        Parameters
        ----------
        queue:
            Optional :class:`TaskQueue` to register the new task with.
        mode:
            Execution mode forwarded to :meth:`TaskItem.set_script_cmd`.
        **kwargs:
            Extra parameters passed to the script command.

        Returns
        -------
        TaskItem
            The newly created (not yet started) task item.
        """
        item = TaskItem.create(self.script , source = 'app' , queue=queue)
        params = kwargs | {'task_id': item.id , 'source': item.source}
        item.set_script_cmd(self.script, params, mode)
        return item

    def preview_cmd(self , mode: Literal['shell', 'os'] = 'shell' , **kwargs) -> str:
        """Return the command string that *would* be used to run the script, without executing it.

        Parameters
        ----------
        mode:
            Execution mode forwarded to :meth:`TaskItem.preview_cmd`.
        **kwargs:
            Extra parameters included in the preview command.

        Returns
        -------
        str
            The formatted command string.
        """
        return TaskItem.preview_cmd(self.script , 'app' , mode , **kwargs)
