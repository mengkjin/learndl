import shutil
from pathlib import Path

from src.proj.env import PATH

class SharedSync:
    """Shared folder sync manager, used to sync the shared folder to the local folder and vice versa"""
    SYNC_DIRS = ['schedule_model']

    @classmethod
    def shared_files(cls) -> list[Path]:
        """get the files to sync"""
        files = []
        if PATH.share_folder is None:
            return files
        for sync_dir in cls.SYNC_DIRS:
            for file in PATH.share_folder.joinpath(sync_dir).rglob('*'):
                if file.is_file() and not file.name.startswith('.'):
                    files.append(file)
        return files

    @classmethod
    def local_files(cls) -> list[Path]:
        """get the files to sync"""
        files = []
        for sync_dir in cls.SYNC_DIRS:
            for file in PATH.local_share_folder.joinpath(sync_dir).rglob('*'):
                if file.is_file() and not file.name.startswith('.'):
                    files.append(file)
        return files

    @classmethod
    def shared_to_local(cls , path : Path) -> Path | None:
        """get the target path of the shared folder"""
        if PATH.share_folder is None:
            return None
        return PATH.local_share_folder.joinpath(path.relative_to(PATH.share_folder))

    @classmethod
    def local_to_shared(cls , path : Path) -> Path | None:
        """get the target path of the shared folder"""
        if PATH.share_folder is None:
            return None
        return PATH.share_folder.joinpath(path.relative_to(PATH.local_share_folder))

    @classmethod 
    def sync(cls):
        """sync all designated files"""
        for shared_file in cls.shared_files():
            local_path = cls.shared_to_local(shared_file)
            if local_path is not None and not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(shared_file, local_path)

        for local_file in cls.local_files():
            shared_path = cls.local_to_shared(local_file)
            if shared_path is not None and not shared_path.exists():
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(local_file, shared_path)