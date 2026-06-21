"""
Direct calls related to file management of this project.
1. git management
2. model / log / config / data files deletion / archiving / resuming / modification
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime , timedelta
from pathlib import Path

from src.proj import MACHINE , PATH , Logger
from src.proj.util.functional.ask import AskFor
from src.api.util.direct_call import DirectCall

__all__ = [
    'ClearOutdatedCatcherLogs' ,
    # 'ArchiveCurrentModel' , 'ResumeArchivedModel' ,
    'ReplaceWeztermConfig' ,
]  
         
# %% model files related operations ------------------------------------------------------------
# class ArchiveCurrentModel(DirectCall):
#     """Archive current model(s) in the model directory."""
#     category = 'Files'
#     def run(self) -> None:
#         from src.res.model.util import ModelPath
        
#         roots = [PATH.model_nn , PATH.model_boost , PATH.model_factor]
#         for loop_flag in AskFor.LoopTillExit(message = f'Do you want to archive more models?'):
#             paths = [(root.name , path , PATH.relative(path)) for root in roots for path in root.iterdir() if path.is_dir()]
#             if not paths:
#                 Logger.note('No models found in the model directory.')
#                 return
#             Logger.note(f'There are {len(paths)} models currently in the model directory...')
#             last_root = None
#             for i , (root_name , path , rel_path) in enumerate(paths):
#                 if last_root is None or last_root != root_name:
#                     Logger.note(f'{root_name.upper()} models:')
#                     last_root = root_name
#                 Logger.note(f'{i+1:02d}. {PATH.relative(rel_path)}' , indent = 1)
#             flag = AskFor.Selections([rel_path for _, _, rel_path in paths] , multiple=True , title = f'Which model to archive?')
#             if loop_flag.set_flag(flag):
#                 [ModelPath(paths[i - 1][1]).move_to_archive() for i in flag.results]

# class ResumeArchivedModel(DirectCall):
#     """Resume archived model(s) from the archive directory."""
#     category = 'Files'
#     def run(self) -> None:
#         from src.res.model.util import ModelPath
        
#         for loop_flag in AskFor.LoopTillExit(message = f'Do you want to resume more models?'):
#             archive_paths = [path for path in PATH.model_archive.iterdir() if path.is_dir()]
#             if not archive_paths:
#                 Logger.note('No models found in the archive directory.')
#                 return
#             flag = AskFor.Options(
#                 archive_paths , confirm = False , multiple=True , 
#                 title = f'Which model to resume from archive?')
#             if loop_flag.set_flag(flag):
#                 [ModelPath.resume_from_archive(path.name) for path in flag.results]

class ModelArchiveOperations(DirectCall):
    """Manage model archive , including archive / resume / rename / packing."""
    category = 'Files'

    @classmethod
    def _iter_current_models(cls , parents : tuple[str,...]= ('nn' , 'boost' , 'factor') , print_paths : bool = True) -> list[Path]:
        roots : list[Path] = [getattr(PATH, f'model_{parent}') for parent in parents]
        paths = [(root.name , path) for root in roots for path in root.iterdir() if path.is_dir()]
        if paths:
            Logger.note(f'There are {len(paths)} models currently in {parents} directories...')
        else:
            Logger.note(f'No models found in {parents} directories.')
        if print_paths and paths:
            last_root = None
            for i , (root_name , path) in enumerate(paths):
                if last_root is None or last_root != root_name:
                    Logger.note(f'{root_name.upper()} models:')
                    last_root = root_name
                Logger.note(f'{i+1:02d}. {PATH.relative(path)}' , indent = 1)
        return [path for _, path in paths]

    @classmethod
    def archive_current_models(cls):
        from src.res.model.util import ModelPath
        title = f'Which model to archive?'
        paths = cls._iter_current_models(parents = ('nn' , 'boost' , 'factor'))
        if not paths:
            return AskFor.flag('abort')
        elif flag := AskFor.Options(paths , confirm = False , multiple=True , title = title , print_options = False):
            [ModelPath(path).move_to_archive() for path in flag.results]
        return flag

    @classmethod
    def resume_archived_models(cls):
        from src.res.model.util import ModelPath
        title = f'Which model to resume from archive?'
        paths = cls._iter_current_models(parents = ('archive' ,))
        if not paths:
            return AskFor.flag('abort')
        elif flag := AskFor.Options(paths , confirm = False , multiple=True , title = title , print_options = False):
            [ModelPath.resume_from_archive(path.name) for path in flag.results]
        return flag

    @classmethod
    def reindex_all_current_models(cls):
        from src.res.model.util import ModelPath
        paths = cls._iter_current_models(parents = ('nn' , 'boost'))
        if not paths:
            return AskFor.flag('abort')
        else:
            [ModelPath(path).auto_reindex() for path in paths]
            return AskFor.flag('yes')

    @classmethod
    def rename_current_models(cls):
        from src.res.model.util import ModelPath
        title = f'Which model to rename?'
        paths = cls._iter_current_models(parents = ('nn' , 'boost'))
        if not paths:
            return AskFor.flag('abort')
        elif flag := AskFor.Options(paths , confirm = False , multiple=False , title = title , print_options = False):
            assert flag.result is not None , 'No model selected'
            name_flag = AskFor.string(title = f'Enter the new name for {flag.result}?')
            if name_flag and name_flag.result:
                ModelPath(flag.result).rename(name_flag.result)
        return flag

    @classmethod
    def _menu_for_operations(cls):
        options = [
            'Archive current models' , 
            'Resume archived models' , 
            'Reindex all current models' , 
            'Rename current models' ,
        ]
        flag = AskFor.Options(options , confirm = False , multiple = False , title = f'What model archive operations to conduct?')
        if flag.result is None:
            return flag
        selection = options.index(flag.result)
        if selection == 0:
            return cls.archive_current_models()
        elif selection == 1:
            return cls.resume_archived_models()
        elif selection == 2:
            return cls.reindex_all_current_models()
        elif selection == 3:
            return cls.rename_current_models()
        else:
            raise ValueError(f'Invalid operation: {flag.result}')

    def run(self) -> None:
        for _ in AskFor.LoopTillExit(False , message = f'Do you want to conduct more model archive operations?'):
            self._menu_for_operations()


# %% log files related operations ------------------------------------------------------------

class ClearOutdatedCatcherLogs(DirectCall):
    """Clear outdated catcher logs."""
    category = 'Files'
    def __init__(self , days_ago : int = 30 , **kwargs):
        assert days_ago > 2 , 'days_ago must be greater than 2'
        self.kwargs = kwargs | {'days_ago': days_ago}
    @property
    def days_ago(self) -> int:
        return self.kwargs['days_ago']
    @classmethod
    def get_description(cls, days_ago : int = 30, **kwargs) -> str:
        return f'Clear outdated catcher logs modified more than {days_ago} days ago. '

    def run(self) -> None:
        root = PATH.logs.joinpath('catcher')
        cleared_counts : dict[str, int] = defaultdict(int)
        if not root.exists():
            return
        for sub_catcher in root.iterdir():
            if sub_catcher.name.startswith('.'):
                continue
            assert sub_catcher.is_dir() , f'{sub_catcher} is not a directory'
            for log_file in sub_catcher.rglob('*'):
                if log_file.is_file() and log_file.stat().st_mtime < (datetime.now() - timedelta(days=self.days_ago)).timestamp():
                    cleared_counts[str(PATH.relative(log_file.parent))] += 1
                    log_file.unlink()
                if log_file.is_dir() and not list(log_file.glob('*')):
                    log_file.rmdir()
                    cleared_counts['empty_dir'] += 1

        Logger.stdout_pairs(cleared_counts , title = f'Cleared {sum(cleared_counts.values())} log files in total, details:')

# %% other files related operations ------------------------------------------------------------
class ReplaceWeztermConfig(DirectCall):
    """Replace the wezterm config file with the latest one."""
    category = 'Files'
    def run(self) -> None:
        
        from pathlib import Path
        from src.proj import PATH
        wezterm_config = PATH.template.joinpath('lua' , 'wezterm.lua')
        if not wezterm_config.exists():
            Logger.error(f"Wezterm config file not found: {wezterm_config}")
            return
        if MACHINE.is_windows:
            target_config = Path.home() / ".config" / "wezterm" / "wezterm.lua"
        else:
            target_config = Path('~/.config/wezterm/wezterm.lua').expanduser()
        if not target_config.exists():
            target_config.parent.mkdir(parents=True, exist_ok=True)
            target_config.touch()
            Logger.success(f"Target config file created: {target_config}")

        content = wezterm_config.read_text()

        if MACHINE.is_macos:
            replacements = {
                'FONT_SIZE' : 12 ,
            }
        elif MACHINE.is_linux:
            replacements = {
                'FONT_SIZE' : 9,
            }
        else:
            replacements = {
                'FONT_SIZE' : 9
            }
        
        for key, value in replacements.items():
            content = content.replace(f'${key}$', str(value))

        if '$' in content:
            Logger.error(f"Placeholder not replaced in content:")
            for i , line in enumerate(content.split('\n')):
                if '$' in line:
                    Logger.error(f"line{i+1: 3d} >> {line}")
            return
        target_config.write_text(content)
        Logger.success(f"Wezterm config file replaced with the latest one.")