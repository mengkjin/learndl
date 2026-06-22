"""
Direct calls related to file management of this project.
1. git management
2. model / log / config / data files deletion / archiving / resuming / modification
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime , timedelta
from pathlib import Path

from src.proj import MACHINE , PATH , Logger , Proj
from src.proj.util.functional.ask import AskFor
from src.api.util.direct_call import DirectCall

__all__ = [
    'ClearOutdatedCatcherLogs',
    'ReplaceWeztermConfig' ,
    'ProjectAutoFix' ,
]  

class ModelArchiveOperations(DirectCall):
    """Manage model archive , including archive / resume / rename / packing."""
    category = 'Files'

    @classmethod
    def _iter_current_models(cls , parents : tuple[str,...]= ('nn' , 'boost' , 'factor') , print_paths : bool = True) -> list[Path]:
        roots : list[Path] = [getattr(PATH, f'model_{parent}') for parent in parents]
        paths = [(root.name , path) for root in roots for path in sorted(root.iterdir()) if path.is_dir()]
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
            return AskFor.flag('invalid')
        elif flag := AskFor.Options(paths , confirm = False , multiple=True , title = title , print_options = False):
            [ModelPath(path).move_to_archive() for path in flag.results]
        return flag

    @classmethod
    def resume_archived_models(cls):
        from src.res.model.util import ModelPath
        title = f'Which model to resume from archive?'
        paths = cls._iter_current_models(parents = ('archive' ,))
        if paths:
            flag = AskFor.Options(paths , confirm = False , multiple=True , title = title , print_options = False)
            if flag.results:
                [ModelPath.resume_from_archive(path.name) for path in flag.results]
        return AskFor.flag()

    @classmethod
    def reindex_all_current_models(cls):
        from src.res.model.util import ModelPath
        paths = cls._iter_current_models(parents = ('nn' , 'boost') , print_paths = False)
        reindex_results : dict[str, str] = {}
        with Proj.silence:
            for path in paths:
                new_model_path = ModelPath(path).auto_reindex()
                if new_model_path is None:
                    reindex_results[str(PATH.relative(path))] = 'Remains!'
                else:
                    reindex_results[str(PATH.relative(path))] = str(PATH.relative(new_model_path.base))
        Logger.stdout_pairs(reindex_results , title = f'Reindex results:')
        return AskFor.flag()

    @classmethod
    def rename_current_models(cls):
        from src.res.model.util import ModelPath
        title = f'Which model to rename?'
        paths = cls._iter_current_models(parents = ('nn' , 'boost'))
        if paths:
            flag = AskFor.Options(paths , confirm = False , multiple=False , title = title , print_options = False)
            assert flag.result is not None , 'No model selected'
            name_flag = AskFor.string(title = f'Enter the new name for {flag.result}?')
            if name_flag and name_flag.result:
                ModelPath(flag.result).rename(name_flag.result)
        return AskFor.flag()

    @classmethod
    def show_model_creation_time(cls):
        from src.res.model.util import ModelPath
        paths = cls._iter_current_models(parents = ('nn' , 'boost' , 'factor') , print_paths = False)
        results = {
            str(PATH.relative(path)) : ModelPath(path).get_creation_time() for path in paths
        }
        Logger.stdout_pairs(results , title = f'Model creation time:')
        return AskFor.flag()

    @classmethod
    def _menu_for_operations(cls):
        options = [
            'Archive current models' , 
            'Resume archived models' , 
            'Reindex all current models' , 
            'Rename current models' ,
            'Show model creation time' ,
        ]
        flag = AskFor.Options(options , confirm = False , multiple = False , title = f'What model archive operations to conduct?')
        if flag.result is None:
            return flag
        selection = options.index(flag.result)
        if selection == 0:
            flag = cls.archive_current_models()
        elif selection == 1:
            flag = cls.resume_archived_models()
        elif selection == 2:
            flag = cls.reindex_all_current_models()
        elif selection == 3:
            flag = cls.rename_current_models()
        elif selection == 4:
            flag = cls.show_model_creation_time()
        else:
            raise ValueError(f'Invalid operation: {flag.result}')
        return flag

    def run(self) -> None:
        for loop in AskFor.LoopTillExit(False):
            loop.set_flag(self._menu_for_operations())


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

class CheckFixAllConfigFiles(DirectCall):
    """Check and auto modify all config files."""
    category = 'Research'
    def run(self) -> None:
        from src.res.model.util.config.inspector import ModelConfigsInspector
        from src.res.model.util.config.modifier import ModelConfigsBatchModifier
        from src.proj import Logger
        Logger.stdout('Checking all config files...')
        modifier = ModelConfigsBatchModifier()
        modifier.batch_modify()
        inspecter = ModelConfigsInspector()
        inspecter.inspect_key_values()
        Logger.success('All config files checked.')

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

class ProjectAutoFix(DirectCall):
    """Apply the project patches."""
    category = 'Files'
    @classmethod
    def _check_all_config_files(cls):
        CheckFixAllConfigFiles.go()
        return AskFor.flag()
    @classmethod
    def _replace_wezterm_config(cls):
        ReplaceWeztermConfig.go()
        return AskFor.flag()
    @classmethod
    def _clear_outdated_catcher_logs(cls):
        ClearOutdatedCatcherLogs.go()
        return AskFor.flag()
    @classmethod
    def _menu_for_operations(cls):
        options = [
            'All AutoFixes' , 
            'Check & Fix all config files' , 
            'Replace wezterm config' , 
            'Clear outdated catcher logs' ,
        ]
        flag = AskFor.Options(options , confirm = False , multiple = False , title = f'What project auto fixes to conduct?')
        if flag.result is None:
            return flag
        selection = options.index(flag.result)
        if selection == 0:
            cls._check_all_config_files()
            cls._replace_wezterm_config()
            cls._clear_outdated_catcher_logs()
            flag = AskFor.flag('exit')
        elif selection == 1:
            flag = cls._check_all_config_files()
        elif selection == 2:
            flag = cls._replace_wezterm_config()
        elif selection == 3:
            flag = cls._clear_outdated_catcher_logs()
        else:
            raise ValueError(f'Invalid operation: {flag.result}')
        return flag

    def run(self) -> None:
        for loop in AskFor.LoopTillExit(False):
            loop.set_flag(self._menu_for_operations())
