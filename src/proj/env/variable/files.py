"""Project-wide log file handle and named, thread-safe file lists (email attachments, exit files)."""

from __future__ import annotations
import io
from dataclasses import dataclass
from pathlib import Path

from src.proj.core import stderr , strPath

__all__ = ['EmailAttachment' , 'LogWriterFile' , 'UniqueFileList']

def validate_attachment_filename(filename : str) -> str:
    """Require a single path segment so the mail name cannot change directories."""
    if not filename or filename in {'.' , '..'}:
        raise ValueError(f'attachment filename is empty: {filename!r}')
    if any(ch in filename for ch in '\\/\r\n\x00') or Path(filename).name != filename:
        raise ValueError(f'attachment filename must be a bare file name, got {filename!r}')
    return filename

@dataclass(frozen = True , slots = True)
class EmailAttachment:
    """Local file plus the name the recipient sees.

    ``filename is None`` keeps the on-disk name. The local path is never renamed.
    """
    path : Path
    filename : str | None = None

    def __post_init__(self) -> None:
        path = self.path if isinstance(self.path , Path) else Path(self.path)
        object.__setattr__(self , 'path' , path)
        if not self.filename:
            object.__setattr__(self , 'filename' , None)
            return
        object.__setattr__(self , 'filename' , validate_attachment_filename(self.filename))

    def __str__(self) -> str:
        return str(self.path)

    def __fspath__(self) -> str:
        return str(self.path)

    @property
    def attachment_name(self) -> str:
        """Recipient-facing file name."""
        return self.filename or self.path.name

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
    _aliases : dict[str , dict[Path , str]] = {}
    def __init__(self , name : str):
        """Register a named list stored in the class-level ``_file_lists`` map."""
        import threading
        self.name = name
        self.lock = threading.Lock()
        self._file_lists[self.name] = []
        self._aliases[self.name] = {}
        self.ban_patterns = []

    def alter1(self , *args , **kwargs):
        """Forward to ``Logger.alert1`` (lazy import to avoid cycles)."""
        from src.proj.log import Logger
        Logger.alert1(*args , **kwargs)

    @property
    def file_list(self):
        """Mutable list of ``Path`` for this instance's ``name``."""
        return self._file_lists[self.name]

    def _alias_map(self) -> dict[Path , str]:
        return self._aliases[self.name]

    def pop_all(self) -> list[EmailAttachment]:
        """Remove and return queued files; clears paths and mail names under lock."""
        with self.lock:
            aliases = self._alias_map()
            queued = [EmailAttachment(path , aliases.get(path)) for path in self.file_list]
            self.file_list.clear()
            aliases.clear()
            return queued

    def append(self , file : strPath , * , filename : str | None = None):
        """Append a path if it is new and not banned.

        ``filename`` is the email attachment name. Omit it to use the local name.
        The file on disk is not renamed.
        """
        alias = validate_attachment_filename(filename) if filename else None
        with self.lock:
            file = Path(file)
            if file in self.file_list:
                return
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to append {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                return
            self.file_list.append(file)
            if alias:
                self._alias_map()[file] = alias

    def extend(self , *files : strPath):
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
    
    def insert(self , index : int , file : strPath):
        """Insert at ``index``; if ``file`` already present, remove old occurrence first."""
        with self.lock:
            file = Path(file)
            if any(pattern in str(file) for pattern in self.ban_patterns):
                self.alter1(f'Fail to insert {file} to {self.name} due to banned patterns!' , vb_level = 'max')
                return
            if file in self.file_list:
                self.file_list.remove(file)
            self.file_list.insert(index , file)

    def remove(self , file : strPath):
        """Remove ``file`` from the list."""
        with self.lock:
            file = Path(file)
            self.file_list.remove(file)
            self._alias_map().pop(file , None)

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
            aliases = self._alias_map()
            for file in self.file_list[:]:
                if any(pattern in str(file) for pattern in patterns):
                    self.file_list.remove(file)
                    aliases.pop(file , None)
                    self.alter1(f'Removed file {file} from {self.name} due to banned patterns!' , vb_level = 'max')