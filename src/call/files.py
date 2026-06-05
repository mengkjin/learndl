"""
Function calls related to file management of this project.
1. git management
2. model / log / config / data files deletion / archiving / resuming / modification
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import datetime , timedelta
from typing import Literal

from src.proj import MACHINE , PATH , Logger

# %% project code related operations
def git_commit_sync(commit_message : str = 'auto commit' , verbose_level : Literal[0,1,2] = 1) -> None:
    """
    commit and sync the code
    Args:
        verbose_level: 0: no output, 1: only success, 2: all output
    """
    prefixes = f'{MACHINE.name} >> {datetime.now().strftime('%Y%m%d')} : {commit_message}'
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', prefixes], check=True)
    subprocess.run(['git', 'pull', '--rebase'], check=True)
    subprocess.run(['git', 'push'], check=True)
    if verbose_level >= 1:
        Logger.success(f"Finish commit and sync")
    if verbose_level >= 2:
        Logger.stdout(f"Commit message: {prefixes}")

def clear_git_pull(verbose_level : Literal[0,1,2] = 1) -> None:
    """
    clear local changes and pull latest code
    Args:
        verbose_level: 0: no output, 1: only success, 2: all output
    """
    import shutil
    assert not MACHINE.platform_coding, "Git Pull is not available on coding platform"

    # clean local changes
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
    subprocess.run(['git', 'clean', '-fd'], check=True)
    if verbose_level >= 1:
        Logger.success(f"Clean local changes done")
    
    # pull latest code
    result = subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True)
    if verbose_level >= 1:
        Logger.success(f"Pull latest code done")
    if verbose_level >= 2:
        Logger.stdout(result.stdout)
    
    empty_folders = []
    for folder in [*PATH.main.joinpath('src').rglob('*/') , *PATH.main.joinpath('configs').rglob('*/')][::-1]:
        if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
            subfiles = [x for x in folder.rglob('*') if x.is_file()]
            if not len(subfiles):
                if verbose_level >= 2:
                    Logger.stdout(f"Removing empty folder: {folder}" , indent = 1)
                folder.rmdir()
                empty_folders.append(folder)
            elif all([x.suffix == '.pyc' for x in subfiles]):
                if verbose_level >= 2:
                    Logger.stdout(f"Removing folder with only pyc files: {folder}" , indent = 1)
                shutil.rmtree(folder)
                empty_folders.append(folder)
            else:
                Logger.error(f"Error removing folder: {folder}:")
                Logger.error(f"Subfiles: {subfiles}")
    if verbose_level >= 1:
        Logger.success(f"Removed {len(empty_folders)} empty folders done")

def check_code_issues():
    """Check if the future annotations are used in the project code."""
    from src.proj.util.functional.ask import AskFor
    from typing import Callable
    def _check_lacking_future_annotations(context : str) -> bool:
        return 'from __future__ import annotations' not in context
    def _check_py_file_has_no_module_docstring(context : str) -> bool:
        lines = context.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('"""') or line.startswith("'''"):
                return False
            return True
        return True
    check_list : dict[str , Callable[[str], bool]] = {
        'lacking future annotations' : _check_lacking_future_annotations,
        'py file has no module docstring' : _check_py_file_has_no_module_docstring,
    }
    flag = AskFor.Options(list(check_list.keys()) , confirm = False , multiple = False , title = f'What code issues to check?')
    if not flag.yes:
        return
    predicate = check_list[flag.result]
    path = PATH.main.joinpath('src')
    Logger.note(f"Code issue {flag.result} found in path:")
    for file in path.glob('**/*.py'):
        if file.name.startswith('__init__'):
            continue
        context = file.read_text()
        if predicate(context):
            Logger.stdout(file , indent = 1)

# %% model files related operations ------------------------------------------------------------

def archive_current_model() -> None:
    """Archive the model."""
    
    from src.res.model.util import ModelPath
    from src.proj.util.functional.ask import AskFor
    roots = [PATH.model_nn , PATH.model_boost , PATH.model_st , PATH.model_factor]

    while True:
        model_paths = [(root.name , path) for root in roots for path in root.iterdir() if path.is_dir()]
        if not model_paths:
            Logger.note('No models found in the model directory.')
            return
        Logger.note(f'There are {len(model_paths)} models currently in the model directory...')
        last_root = None
        for i , (root_name , model_path) in enumerate(model_paths):
            if last_root is None or last_root != root_name:
                Logger.note(f'{root_name.upper()} models:')
                last_root = root_name
            Logger.note(f'{i+1:02d}. {model_path.relative_to(PATH.model)}' , indent = 1)
        flag = AskFor.Selections(len(model_paths) , multiple=True , title = f'Which model to archive?')
        if flag.no:
            return
        if flag.yes:
            for i in flag.result:
                ModelPath(model_paths[i - 1][1]).move_to_archive()
        flag = AskFor.Retry(f'Do you want to archive more models?')
        if flag.no:
            return

def resume_archived_model() -> None:
    """Resume the model."""
    from src.res.model.util import ModelPath
    from src.proj.util.functional.ask import AskFor
    
    while True:
        archive_paths = [path for path in PATH.model_archive.iterdir() if path.is_dir()]
        if not archive_paths:
            Logger.note('No models found in the archive directory.')
            return
        flag = AskFor.Options(archive_paths , confirm = False , multiple=True , title = f'Which model to resume from archive?')
        if flag.no:
            return
        if flag.yes:
            for path in flag.result:
                ModelPath.resume_from_archive(path.name)
        flag = AskFor.Retry(title = f'Do you want to resume more models?')
        if flag.no:
            return

# %% log files related operations ------------------------------------------------------------
def clear_outdated_catcher_logs(days_ago : int = 30): 
    assert days_ago > 2 , 'days_ago must be greater than 2'
    root = PATH.logs.joinpath('catcher')
    cleared_counts : dict[str, int] = defaultdict(int)
    for sub_catcher in root.iterdir():
        if sub_catcher.name.startswith('.'):
            continue
        assert sub_catcher.is_dir() , f'{sub_catcher} is not a directory'
        for log_file in sub_catcher.rglob('*'):
            if log_file.is_file() and log_file.stat().st_mtime < (datetime.now() - timedelta(days=days_ago)).timestamp():
                cleared_counts[str(log_file.parent.relative_to(root))] += 1
                log_file.unlink()
            if log_file.is_dir() and not list(log_file.glob('*')):
                log_file.rmdir()
                cleared_counts['empty_dir'] += 1

    Logger.stdout_pairs(cleared_counts , title = f'Cleared {sum(cleared_counts.values())} log files in total, details:')

# %% config files related operations ------------------------------------------------------------
def check_all_config_files():     
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
def replace_wezterm_config():
    """Replace the wezterm config file with the latest one."""
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