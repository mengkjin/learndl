import io
from pathlib import Path
import threading

from src.proj.abc import stderr

class LogWriterFile:
    def __init__(self):
        self.value = None

    def __set__(self , instance, value):
        assert value is None or isinstance(value , io.TextIOWrapper) , f'value is not a {io.TextIOWrapper} instance: {type(value)} , cannot be set to {instance.__name__}.log_file'
        if self.value is None:
            stderr(f'Project Log File Reset to None' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Log File Set to a new file : {value.name}' , color = 'lightred' , bold = True)
        self.value = value

    def __get__(self , instance, owner):
        return self.value

class UniqueFileList:
    _file_lists : dict[str , list[Path]] = {}
    def __init__(self , name : str):
        self.name = name
        self.lock = threading.Lock()
        self._file_lists[self.name] = []
        self.ban_patterns = []

    def alter1(self , *args , **kwargs):
        if not hasattr(self , '_logger'):
            from src.proj.log import Logger
            self._logger = Logger
        self._logger.alert1(*args , **kwargs)

    @property
    def file_list(self):
        return self._file_lists[self.name]

    def pop_all(self):
        with self.lock:
            files = self.file_list[:]
            self.file_list.clear()
            return files

    def append(self , file : Path | str):
        with self.lock:
            file = Path(file)
            if file in self.file_list:
                return
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to append {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                return
            self.file_list.append(file)

    def extend(self , *files : Path | str):
        with self.lock:
            for file in files:
                file = Path(file)
                if file in self.file_list: 
                    continue
                if any(pattern in str(file) for pattern in self.ban_patterns):
                    self.alter1(f'Fail to append {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                    continue
                self.file_list.append(file)
    
    def insert(self , index : int , file : Path | str):
        with self.lock:
            file = Path(file)
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to insert {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                return
            if file in self.file_list:
                self.file_list.remove(file)
            self.file_list.insert(index , file)

    def remove(self , file : Path | str):
        with self.lock:
            self.file_list.remove(Path(file))

    def ban(self , *patterns : str):
        with self.lock:
            self.ban_patterns.extend(patterns)

    def unban(self , *patterns : str):
        with self.lock:
            self.ban_patterns = [pattern for pattern in self.ban_patterns if pattern not in patterns]

    def exclude(self , *patterns : str):
        with self.lock:
            for file in self.file_list[:]:
                if any(pattern in str(file) for pattern in patterns):
                    self.file_list.remove(file)
                    self.alter1(f'Removed file {file} from {self.name} due to banned patterns!' , vb_level = 'max')

LOG_WRITER = LogWriterFile()
EMAIL_ATTACHMENTS = UniqueFileList('email_attachments')
EXIT_FILES = UniqueFileList('exit_files')