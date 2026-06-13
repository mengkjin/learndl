"""
Direct calls related to file management of this project.
1. git management
2. model / log / config / data files deletion / archiving / resuming / modification
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import datetime , timedelta
from pathlib import Path
from typing import Callable

from src.proj import MACHINE , PATH , Logger , Base
from src.call.basic import DirectCall

__all__ = [
    'GitCommitSync' , 'GitClearPull' , 
    'RunRuffPyrightCheck' , 'CheckCodeIssues' ,
    'ClearOutdatedCatcherLogs' ,
    'ArchiveCurrentModel' , 'ResumeArchivedModel' ,
    'CheckAllConfigFiles' ,
    'ReplaceWeztermConfig' ,
]

# %% project code related operations
class GitCommitSync(DirectCall):
    """Commit and sync the code. Args: verbose_level: 0: no output, 1: only success, 2: all output"""
    category = 'Files'
    def __init__(self , commit_message : str = 'auto commit' , verbose_level : Base.lit._0_3 = 1 , **kwargs):
        self.kwargs = kwargs | {'commit_message': commit_message, 'verbose_level': verbose_level}
    @property
    def commit_message(self) -> str:
        return self.kwargs['commit_message']
    @property
    def verbose_level(self) -> Base.lit._0_3:
        return self.kwargs['verbose_level']
    def run(self) -> None:
        prefixes = f'{MACHINE.name} >> {datetime.now().strftime('%Y%m%d')} : {self.commit_message}'
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', prefixes], check=True)
        subprocess.run(['git', 'pull', '--rebase'], check=True)
        subprocess.run(['git', 'push'], check=True)
        if self.verbose_level >= 1:
            Logger.success(f"Finish commit and sync")
        if self.verbose_level >= 2:
            Logger.stdout(f"Commit message: {prefixes}")

class GitClearPull(DirectCall):
    """Clear local changes and pull latest code. Args: verbose_level: 0: no output, 1: only success, 2: all output"""
    category = 'Files'
    def __init__(self , verbose_level : Base.lit._0_3 = 1 , **kwargs):
        self.kwargs = kwargs | {'verbose_level': verbose_level}
    @property
    def verbose_level(self) -> Base.lit._0_3:
        return self.kwargs['verbose_level']
    def run(self) -> None:
        import shutil
        assert not MACHINE.platform_coding, "Git Pull is not available on coding platform"

        # clean local changes
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
        subprocess.run(['git', 'clean', '-fd'], check=True)
        if self.verbose_level >= 1:
            Logger.success(f"Clean local changes done")
        
        # pull latest code
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True)
        if self.verbose_level >= 1:
            Logger.success(f"Pull latest code done")
        if self.verbose_level >= 2:
            Logger.stdout(result.stdout)
        
        empty_folders = []
        for folder in [*PATH.main.joinpath('src').rglob('*/') , *PATH.main.joinpath('configs').rglob('*/')][::-1]:
            if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
                subfiles = [x for x in folder.rglob('*') if x.is_file()]
                if not len(subfiles):
                    if self.verbose_level >= 2:
                        Logger.stdout(f"Removing empty folder: {folder}" , indent = 1)
                    folder.rmdir()
                    empty_folders.append(folder)
                elif all([x.suffix == '.pyc' for x in subfiles]):
                    if self.verbose_level >= 2:
                        Logger.stdout(f"Removing folder with only pyc files: {folder}" , indent = 1)
                    shutil.rmtree(folder)
                    empty_folders.append(folder)
                else:
                    Logger.error(f"Error removing folder: {folder}:")
                    Logger.error(f"Subfiles: {subfiles}")
        if self.verbose_level >= 1:
            Logger.success(f"Removed {len(empty_folders)} empty folders done")

class RunRuffPyrightCheck(DirectCall):
    """Run ruff pyright check on the project code. """
    category = 'Files'
    @classmethod
    def check_args(cls , check_name : str) -> list[str]:
        if check_name == 'ruff':
            return ['ruff' , 'check' , 'src' , 'scripts']
        elif check_name == 'pyright':
            return ['pyright' , 'src' , 'scripts']
        else:
            raise ValueError(f"Invalid check name: {check_name}")
    def run(self) -> None:
        for check_name in ['ruff' , 'pyright']:
            Logger.note(f"Running {check_name} check on the project code...")
            ret = subprocess.run(self.check_args(check_name) , capture_output=True, text=True, check=False)
            Logger.success(f"{check_name.title()} check passed.") if ret.returncode == 0 else Logger.alert2(f"{check_name.title()} check failed: ")
            if ret.stdout.strip():
                Logger.stdout(ret.stdout.strip())
            if ret.stderr.strip():
                Logger.stderr(ret.stderr.strip())

class CheckCodeIssues(DirectCall):
    """Check the code issues in the project code."""
    category = 'Files'
    @staticmethod
    def _skip_path(path : Path , exempt_paths : list[Path] , exempt_names : list[str]) -> bool:
        if not path.is_file() or path.name in exempt_names or any(path.is_relative_to(exempt_path) for exempt_path in exempt_paths):
            return True
        return False
    @classmethod
    def _check_future_annotations_existence(cls ,path : Path) -> bool:
        """Check future annotations existence."""
        if cls._skip_path(path , [] , ['__init__.py' , '__version__.py']):
            return True
        return 'from __future__ import annotations' in path.read_text()
    @classmethod
    def _check___all___existence(cls ,path : Path) -> bool:
        """Check __all__ existence."""
        exempt_paths = [
            PATH.main.joinpath('src' , 'func') ,
            PATH.main.joinpath('src' , 'proj' , 'core' , 'types') ,
        ]
        if cls._skip_path(path , exempt_paths , ['__init__.py' , '__version__.py']):
            return True
        return '__all__' in path.read_text()
    @classmethod
    def _check_module_docstring_existence(cls , path : Path) -> bool:
        """Check module docstring existence."""
        if cls._skip_path(path , [] , ['__init__.py' , '__version__.py']):
            return True
        for line in path.read_text().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            return line.startswith('"""') or line.startswith("'''")
        return False

    @classmethod
    def _run_check(cls , checker : Callable[[Path], bool]) -> None:
        Logger.note(f"{checker.__doc__}...")
        files = [file for file in PATH.main.joinpath('src').rglob('*.py') if not checker(file)]
        if files:
            [Logger.stdout(file , indent = 1) for file in files]
        else:
            Logger.success(f'No files found with the issue {checker.__doc__}.')

    def run(self) -> None:
        from src.proj.util.functional.ask import AskFor
        checkers : list[Callable[[Path], bool]] = [getattr(self.__class__, name) for name in dir(self.__class__) if name.startswith('_check_')]
        for loop_flag in AskFor.LoopTillExit(False , message = f'Do you want to check more code issues?' , max_trials = 100):
            options = ['All checks in one' , 'Run ruff pyright check' , *[checkers.__doc__ for checkers in checkers]]
            flag = AskFor.Options(options , confirm = False , multiple = False , title = f'What code issues to check?')
            if not loop_flag.set_flag(flag):
                continue
            selection = options.index(flag.result)
            if selection == 0:
                for checker in checkers:
                    self._run_check(checker)
                RunRuffPyrightCheck.go()
            elif selection == 1:
                RunRuffPyrightCheck.go()
            else:
                checker = checkers[selection - 2]
                self._run_check(checker)
            
         
# %% model files related operations ------------------------------------------------------------
class ArchiveCurrentModel(DirectCall):
    """Archive current model(s) in the model directory."""
    category = 'Files'
    def run(self) -> None:
        from src.res.model.util import ModelPath
        from src.proj.util.functional.ask import AskFor
        roots = [PATH.model_nn , PATH.model_boost , PATH.model_factor]
        for loop_flag in AskFor.LoopTillExit(message = f'Do you want to archive more models?'):
            paths = [(root.name , path) for root in roots for path in root.iterdir() if path.is_dir()]
            if not paths:
                Logger.note('No models found in the model directory.')
                return
            Logger.note(f'There are {len(paths)} models currently in the model directory...')
            last_root = None
            for i , (root_name , path) in enumerate(paths):
                if last_root is None or last_root != root_name:
                    Logger.note(f'{root_name.upper()} models:')
                    last_root = root_name
                Logger.note(f'{i+1:02d}. {PATH.relative(path)}' , indent = 1)
            flag = AskFor.Selections(len(paths) , multiple=True , title = f'Which model to archive?')
            if not loop_flag.set_flag(flag):
                continue
            for i in flag.result:
                ModelPath(paths[i - 1][1]).move_to_archive()

class ResumeArchivedModel(DirectCall):
    """Resume archived model(s) from the archive directory."""
    category = 'Files'
    def run(self) -> None:
        from src.res.model.util import ModelPath
        from src.proj.util.functional.ask import AskFor
        
        for loop_flag in AskFor.LoopTillExit(message = f'Do you want to resume more models?'):
            archive_paths = [path for path in PATH.model_archive.iterdir() if path.is_dir()]
            if not archive_paths:
                Logger.note('No models found in the archive directory.')
                return
            flag = AskFor.Options(archive_paths , confirm = False , multiple=True , title = f'Which model to resume from archive?')
            if not loop_flag.set_flag(flag):
                continue
            [ModelPath.resume_from_archive(path.name) for path in flag.result]

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

# %% config files related operations ------------------------------------------------------------

class CheckAllConfigFiles(DirectCall):
    """Check and auto modify all config files."""
    category = 'Files'
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