import portalocker , re
from dataclasses import dataclass , field
from datetime import datetime
from pathlib import Path

from src.proj.env import PATH

@dataclass
class LogEntry:
    title : str = ''
    messages : list[str] = field(default_factory=list)
    timestamp : datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.title = self.title.strip()
        self.messages = self.split_messages(self.messages)

    def __bool__(self) -> bool:
        return bool(self.title)

    @classmethod
    def split_messages(cls , messages : list[str] | str | None) -> list[str]:
        if messages is None:
            return []
        elif isinstance(messages , str):
            return [msg for msg in messages.split('\n')]
        elif isinstance(messages , list):
            return [msg for msgs in messages for msg in msgs.split('\n')]
        else:
            raise ValueError(f'Invalid messages type: {type(messages)}')

    def append_message(self , *messages : str):
        new_messages = self.split_messages(list(messages))
        self.messages = self.messages + new_messages

    @property
    def content(self) -> str:
        return '\n'.join(self.content_lines())

    @property
    def title_line(self) -> str:
        return f'{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} >> {self.title.strip()}'

    def content_lines(self , indent : bool = False) -> list[str]:
        if self.messages is None:
            return []
        elif indent:
            return [f'  {msg}' for msg in self.messages]
        else:
            return self.messages

    def to_lines(self) -> list[str]:
        if not self:
            return []
        return [self.title_line , *self.content_lines(indent = True)]

    @classmethod
    def from_args(cls , *args : str) -> 'LogEntry':
        if len(args) == 0:
            return cls()
        long_message = '\n'.join(args)
        title , *messages = long_message.split('\n', 1)
        return cls(title , list(messages))
    
    @classmethod
    def from_lines(cls , lines : list[str] , from_latest : bool = True) -> list['LogEntry']:
        entries : list['LogEntry'] = []
        current_entry : LogEntry | None = None
        for line in lines:
            match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) >> (.*)$', line)
            if match:
                timestr , title = match.groups()
                try:
                    timestamp = datetime.strptime(timestr.strip() , '%Y-%m-%d %H:%M:%S')
                except (IndexError, ValueError) as e:
                    raise ValueError(f'Error {e} parsing time string: {timestr}')
                current_entry = cls(title.strip() , timestamp = timestamp)
                entries.append(current_entry)
            elif current_entry:
                current_entry.append_message(line.removeprefix(' '))
            else:
                # The log file is not started with a title line , ignore the line
                ...
        if from_latest:
            entries = entries[::-1]
        return entries

class LogFile:
    """
    log file class , support log rotation
    """
    def __init__(self , log_file : Path | str , rotate : bool = False , rotation_size_mb : int = 10):
        self.host_file = Path(log_file)
        self.rotate = rotate
        self.rotation_size_mb = rotation_size_mb
        assert self.host_file.suffix == '.log' , f'log file must have .log suffix: {self.host_file}'

        self.host_file.parent.mkdir(parents=True, exist_ok=True)
        self.host_file.touch(exist_ok=True)

        self.check_rotation()

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'LogFile(log_file={self.host_file},rotate={self.rotate})'

    @property
    def log_file(self) -> Path:
        return self.current_file

    
    @property
    def candidates(self) -> list[Path]:
        return [self.host_file] + self.rotation_files

    @property
    def rotation_files(self) -> list[Path]:
        if not hasattr(self , '_rotation_files'):
            self._rotation_files = []
        return self._rotation_files

    @rotation_files.setter
    def rotation_files(self , value : list[Path]):
        self._rotation_files = value

    @property
    def date_suffix(self) -> str:
        return f'.{datetime.now().strftime('%Y-%m-%d')}'

    @property
    def parent(self) -> Path:
        return self.host_file.parent

    @property
    def stem(self) -> str:
        return self.host_file.stem

    @property
    def name(self) -> str:
        return self.host_file.name

    def check_rotation(self):
        if self.rotate:
            self.rotation_files = sorted([path for path in self.parent.glob(f'{self.name}.*')], key=lambda x: x.suffix)
            self.current_file = self.rotation_files[-1] if self.rotation_files else self.host_file
            if self.current_file.stat().st_size > self.rotation_size_mb * 1024 * 1024 and self.current_file.suffix != self.date_suffix:
                self.current_file = self.host_file.with_name(f'{self.name}{self.date_suffix}')
                self.current_file.touch(exist_ok=True)
                self.rotation_files.append(self.current_file)
        else:
            self.current_file = self.host_file

    def unlink(self , confirm : bool = True):
        if confirm:
            value = input(f'Are you sure you want to delete all {self} log files? (y/n): ')
            if value != 'y':
                return
        for log_file in self.rotation_files:
            log_file.unlink(missing_ok=True)
        self.host_file.unlink(missing_ok=True)

    def write_entry(self , entry : LogEntry):
        """write a log entry to log file"""
        if not entry:
            return
        with open(self.log_file , 'a') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            f.write('\n'.join(entry.to_lines()) + '\n')

    def read_entry(self , max_entries : int = -1 , from_latest : bool = True):
        read_files = reversed(self.candidates) if from_latest else self.candidates
        entries : list[LogEntry] = []
        for file in read_files:
            with open(file , 'r') as f:
                lines = f.readlines()
            entries.extend(LogEntry.from_lines(lines , from_latest = from_latest))
            if max_entries > 0 and len(entries) > max_entries:
                break
        if max_entries > 0:
            entries = entries[:max_entries]
        return entries

    def write(self , *args):
        """write a message or a list of messages to log file"""
        self.write_entry(LogEntry.from_args(*args))

    def read(self , max_entries : int = -1 , from_latest : bool = True):
        return self.read_entry(max_entries = max_entries , from_latest = from_latest)

    @classmethod
    def initiate(cls , *args : str , rotate : bool = False , rotation_size_mb : int = 10):
        log_file = PATH.logs.joinpath(*args).with_suffix('.log')
        return cls(log_file , rotate = rotate , rotation_size_mb = rotation_size_mb)
