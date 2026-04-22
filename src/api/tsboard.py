import os , shutil , subprocess
from pathlib import Path
from src.proj import PATH , Logger

def call_tensorboard(log_dir : str | Path):
    """
    Launch TensorBoard via ``uv run tensorboard --logdir`` (blocks until user interrupt).

    Args:
        log_dir: Directory containing event files.
    """
    if isinstance(log_dir , Path):
        log_dir = log_dir.as_posix()
    try:
        subprocess.run(['uv' , 'run' , 'tensorboard' , '--logdir' , log_dir]) 
    except KeyboardInterrupt:
        Logger.alert1("TensorBoard stopped.")
    except Exception as e:
        Logger.error(f"Failed to launch TensorBoard: {e}")
        Logger.print_exc(e)
        raise e

def run_local_tensorboard():
    """
    Start TensorBoard pointing at ``PATH.tensorboard/run`` if logs exist.
    """
    log_dir = PATH.tensorboard.joinpath('run')
    if not log_dir.exists():
        Logger.alert1("No local Tensorboard logs found in run folder")
        return
    call_tensorboard(log_dir.as_posix())

def run_packed_tensorboard():
    """
    Interactively pick a packed ``.tar`` of TensorBoard logs, unpack to temp, and launch.
    """
    packed_tars = list(PATH.tensorboard.glob('*.tar'))
    if len(packed_tars) == 0:
        Logger.alert1("No packed Tensorboard logs found in tar folder")
        return
    Logger.success("Choose from available packed Tensorboard tar files:")
    for i , packed_tar in enumerate(packed_tars):
        Logger.stdout(f"{i + 1:>2}. {packed_tar.name}" , indent = 1 , color = 'lightyellow')
    index = int(input("Enter the number of the packed Tensorboard tar to launch: "))
    assert index >= 1 and index <= len(packed_tars) , f"Invalid index: {index} , must be between 1 and {len(packed_tars)}"
    packed_tar = packed_tars[index - 1]
    log_dir = PATH.tensorboard.joinpath('temp_run')
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(packed_tar, log_dir)
    call_tensorboard(log_dir.as_posix())

def run_trained_models_tensorboard():
    """
    Interactively pick a trained model with preserved tensorboard snapshots and launch TB.
    """
    from src.api.model import ModelAPI
    from src.res.model.util.model_path import ModelPath
    candidates = [ModelPath(model) for model in ModelAPI.available_models(include_short_test = True)]
    candidates = [model for model in candidates if model.snapshot('tensorboard').exists()]
    if len(candidates) == 0:
        Logger.alert1("No available models with tensorboard logs found")
        return
    Logger.success("Choose from available models that have tensorboard logs when no model name is provided:")
    for i , model in enumerate(candidates):
        Logger.stdout(f"{i + 1:>2}. {model.full_name}" , indent = 1 , color = 'lightyellow')
    index = int(input("Enter the number of the model to launch: "))
    assert index >= 1 and index <= len(candidates) , f"Invalid index: {index} , must be between 1 and {len(candidates)}"
    model_name = candidates[index - 1].full_name
    log_dir = ModelPath(model_name).snapshot('tensorboard')
    
    call_tensorboard(log_dir)

class TSBoardAPI:
    @classmethod
    def launch(cls):
        """
        CLI menu to launch TensorBoard for local runs, packed archives, or trained-model logs.

        ``**kwargs`` reserved for future options; currently uses ``stdin`` prompts.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          lock_timeout: 1
          disable_platforms: []
          execution_time: immediate
          memory_usage: medium
        """
        os.chdir(PATH.main)
        Logger.success("Will launch TensorBoard for the following options:")

        options = ["local Tensorboard logs in run folder (lastest training)" , "Packed Tensorboard tar files (all past trainings)" , "Tensorboard logs of Trained Models (preserved training logs)"]
        for i , option in enumerate(options):
            Logger.stdout(f"{i + 1:>2}. {option}" , indent = 1 , color = 'lightyellow')
        index = int(input("Enter the number of the option to launch: "))
        assert index >= 1 and index <= len(options) , f"Invalid index: {index} , must be between 1 and {len(options)}"
        
        if index == 1:
            run_local_tensorboard()
        elif index == 2:
            run_packed_tensorboard()
        elif index == 3:
            run_trained_models_tensorboard()
