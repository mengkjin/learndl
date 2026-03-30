"""Project-wide log file handle and named, thread-safe file lists (email attachments, exit files)."""

import io
from pathlib import Path
import threading

from src.proj.abc import stderr


class LogWriterFile:
    """Descriptor storing the optional main project log stream (``TextIOWrapper``)."""

    def __init__(self):
        self.value = None

    def __set__(self , instance, value):
        assert value is None or isinstance(value , io.TextIOWrapper) , f'value is not a {io.TextIOWrapper} instance: {type(value)} , cannot be set to {instance.__name__}.log_file'
        if value is None:
            stderr(f'Project Log File Reset to None' , color = 'lightred' , bold = True)
        else:
            stderr(f'Project Log File Set to a new file : {value.name}' , color = 'lightred' , bold = True)
        self.value = value

    def __get__(self , instance, owner):
        """Return the current log file handle, or ``None``."""
        return self.value


class UniqueFileList:
    """Thread-safe deduplicated list of paths, keyed by logical name (e.g. email attachments)."""

    _file_lists : dict[str , list[Path]] = {}
    def __init__(self , name : str):
        """Register a named list stored in the class-level ``_file_lists`` map."""
        self.name = name
        self.lock = threading.Lock()
        self._file_lists[self.name] = []
        self.ban_patterns = []

    def alter1(self , *args , **kwargs):
        """Forward to ``Logger.alert1`` (lazy import to avoid cycles)."""
        if not hasattr(self , '_alert1'):
            from src.proj.log import Logger
            self._alert1 = Logger.alert1
        self._alert1(*args , **kwargs)

    @property
    def file_list(self):
        """Mutable list of ``Path`` for this instance's ``name``."""
        return self._file_lists[self.name]

    def pop_all(self):
        """Remove and return all paths; clears the list under lock."""
        with self.lock:
            files = self.file_list[:]
            self.file_list.clear()
            return files

    def append(self , file : Path | str):
        """Append a path if not duplicate and not matching ``ban_patterns``."""
        with self.lock:
            file = Path(file)
            if file in self.file_list:
                return
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to append {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                return
            self.file_list.append(file)

    def extend(self , *files : Path | str):
        """Append multiple paths with the same rules as ``append``."""
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
        """Insert at ``index``; if ``file`` already present, remove old occurrence first."""
        with self.lock:
            file = Path(file)
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to insert {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                return
            if file in self.file_list:
                self.file_list.remove(file)
            self.file_list.insert(index , file)

    def remove(self , file : Path | str):
        """Remove ``file`` from the list."""
        with self.lock:
            self.file_list.remove(Path(file))

    def ban(self , *patterns : str):
        """Substrings; paths containing any pattern are rejected on add."""
        with self.lock:
            self.ban_patterns.extend(patterns)

    def unban(self , *patterns : str):
        """Remove patterns from ``ban_patterns``."""
        with self.lock:
            self.ban_patterns = [pattern for pattern in self.ban_patterns if pattern not in patterns]

    def exclude(self , *patterns : str):
        """Drop existing paths whose string form contains any of ``patterns``."""
        with self.lock:
            for file in self.file_list[:]:
                if any(pattern in str(file) for pattern in patterns):
                    self.file_list.remove(file)
                    self.alter1(f'Removed file {file} from {self.name} due to banned patterns!' , vb_level = 'max')

LOG_WRITER = LogWriterFile()
EMAIL_ATTACHMENTS = UniqueFileList('email_attachments')
EXIT_FILES = UniqueFileList('exit_files')