"""
Function calls for the quick call buttons.
"""
from src.proj import PATH , Logger
from src.proj.util.util_funcs import ask_for_selections , ask_for_retry

def archive_model() -> None:
    """Archive the model."""
    
    from src.res.model.util import ModelPath
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
        flag = ask_for_selections(f'Which model to archive?' , len(model_paths))
        if flag.no:
            return
        if flag.yes:
            for i in flag.result:
                ModelPath(model_paths[i - 1][1]).move_to_archive()
        flag = ask_for_retry(f'Do you want to archive more models?')
        if flag.no:
            return

def resume_model() -> None:
    """Resume the model."""
    from src.res.model.util import ModelPath
    
    while True:
        archive_paths = [path for path in PATH.model_archive.iterdir() if path.is_dir()]
        if not archive_paths:
            Logger.note('No models found in the archive directory.')
            return
        Logger.note(f'There are {len(archive_paths)} models currently in the archive directory...')

        for i , archive_path in enumerate(archive_paths):
            Logger.note(f'{i+1:02d}. {archive_path.relative_to(PATH.model_archive)}' , indent = 1)
        flag = ask_for_selections(f'Which model to resume?' , len(archive_paths))
        if flag.no:
            return
        if flag.yes:
            for i in flag.result:
                ModelPath.resume_from_archive(archive_paths[i - 1].name)
        flag = ask_for_retry(f'Do you want to resume more models?')
        if flag.no:
            return