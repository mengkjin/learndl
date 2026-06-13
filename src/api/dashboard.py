"""
API for launching dashboard of this project.
"""

from __future__ import annotations
import os , shutil , subprocess , webbrowser , time , socket
from pathlib import Path
from src.proj import PATH , Logger , Base

__all__ = ['DashboardAPI' , 'OptunaDBAPI' , 'TSBoardAPI']

def get_free_port(start_port=8080):
    """Finds an available TCP port starting from start_port."""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port  # Port is free!
            except OSError:
                port += 1  # Occupied, try next port

def open_url(host : str , port: int):
    # Wait 1.5 seconds for the server to spin up completely
    time.sleep(1.5)
    webbrowser.open(f"http://{host}:{port}/")

class DashboardAPI:
    """API for launching dashboard of this project."""
    @classmethod
    def optuna_dashboard(cls):
        """
        CLI menu to launch Optuna Dashboard for local runs, saved databases, or trained-model logs.

        ``**kwargs`` reserved for future options; currently uses ``stdin`` prompts.

        [API Interaction]:
          expose: true
          email: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          lock_timeout: 1
          disable_platforms: []
          execution_time: immediate
          memory_usage: medium
        """
        OptunaDBAPI.launch(open_browser = True)

    @classmethod
    def tensorboard(cls):
        """
        CLI menu to launch TensorBoard for local runs, packed archives, or trained-model logs.

        ``**kwargs`` reserved for future options; currently uses ``stdin`` prompts.

        [API Interaction]:
          expose: true
          email: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          lock_timeout: 1
          disable_platforms: []
          execution_time: immediate
          memory_usage: medium
        """
        TSBoardAPI.launch(open_browser = True)

class OptunaDBAPI:
    """API for launching Optuna Dashboard."""
    @classmethod
    def call_optuna_dashboard(cls , db_path : Base.strPath , * , open_browser : bool = False):
        """
        Launch Optuna Dashboard via ``optuna-dashboard`` (blocks until user interrupt).

        Args:
            db_path: Path to the SQLite database file.
        """
        if isinstance(db_path , str):
            db_path = Path(db_path)
        db_path = f'sqlite:///{str(PATH.relative(db_path)).lstrip('/')}'
        try:
            port = get_free_port(8080)
            Logger.stdout(f"Optuna Dashboard will be launched on http://127.0.0.1:{port}/")
            if open_browser:
                import threading
                threading.Thread(target=open_url, args=("127.0.0.1",port)).start()
            subprocess.run(['optuna-dashboard' , db_path , '--port' , str(port)]) 
        except KeyboardInterrupt:
            Logger.alert1("Optuna Dashboard stopped.")
        except Exception as e:
            Logger.error(f"Failed to launch Optuna Dashboard: {e}")
            Logger.print_exc(e)
            raise e

    @classmethod
    def run_latest_optuna_dashboard(cls , * , open_browser : bool = False):
        """
        Start Optuna Dashboard pointing at ``PATH.optuna/run`` if logs exist.
        """
        candidates = list(PATH.optuna.glob('*.sqlite3'))
        if not candidates:
            Logger.alert1("No local Optuna Dashboard database found in optuna folder")
            return
        candidates.sort(key = lambda x: x.stat().st_mtime , reverse = True)
        cls.call_optuna_dashboard(candidates[0] , open_browser = open_browser)

    @classmethod
    def run_saved_optuna_dashboard(cls , * , open_browser : bool = False):
        """
        Interactively pick a saved Optuna Dashboard database and launch it.
        """
        candidates = list(PATH.optuna.glob('*.sqlite3'))
        if not candidates:
            Logger.alert1("No available Optuna Dashboard databases found")
            return
        Logger.success("Choose from available Optuna Dashboard databases:")
        for i , db_path in enumerate(candidates):
            Logger.stdout(f"{i + 1:>2}. {db_path.name}" , indent = 1 , color = 'yellow')
        index = int(input("Enter the number of the Optuna Dashboard database to launch: "))
        assert index >= 1 and index <= len(candidates) , f"Invalid index: {index} , must be between 1 and {len(candidates)}"
        cls.call_optuna_dashboard(candidates[index - 1] , open_browser = open_browser)
        
    @classmethod
    def launch(cls , * , open_browser : bool = False):
        """
        launch Optuna Dashboard for local runs, saved databases.
        """
        os.chdir(PATH.main)
        Logger.success("Will launch Optuna Dashboard for the following options:")

        options = ["Latest Run Optuna Dashboard database" , "Select Saved Optuna Dashboard database"]
        for i , option in enumerate(options):
            Logger.stdout(f"{i + 1:>2}. {option}" , indent = 1 , color = 'yellow')
        index = int(input("Enter the number of the option to launch: "))
        assert index >= 1 and index <= len(options) , f"Invalid index: {index} , must be between 1 and {len(options)}"
        
        if index == 1:
            cls.run_latest_optuna_dashboard(open_browser = open_browser)
        elif index == 2:
            cls.run_saved_optuna_dashboard(open_browser = open_browser)


class TSBoardAPI:
    """API for launching TensorBoard."""
    @classmethod
    def call_tensorboard(cls , log_dir : Base.strPath , * , open_browser : bool = False):
        """
        Launch TensorBoard via ``uv run tensorboard --logdir`` (blocks until user interrupt).

        Args:
            log_dir: Directory containing event files.
        """
        if isinstance(log_dir , Path):
            log_dir = log_dir.as_posix()
        try:
            port = get_free_port(6006)
            Logger.stdout(f"TensorBoard will be launched on http://localhost:{port}/")
            if open_browser:
                import threading
                threading.Thread(target=open_url, args=("localhost",port)).start()
            subprocess.run(['uv' , 'run' , 'tensorboard' , '--logdir' , log_dir , '--port' , str(port)]) 
        except KeyboardInterrupt:
            Logger.alert1("TensorBoard stopped.")
        except Exception as e:
            Logger.error(f"Failed to launch TensorBoard: {e}")
            Logger.print_exc(e)
            raise e

    @classmethod
    def run_local_tensorboard(cls , * , open_browser : bool = False):
        """
        Start TensorBoard pointing at ``PATH.tensorboard/run`` if logs exist.
        """
        log_dir = PATH.tsboard.joinpath('run')
        if not log_dir.exists():
            Logger.alert1("No local Tensorboard logs found in run folder")
            return
        cls.call_tensorboard(log_dir , open_browser = open_browser)

    @classmethod
    def run_packed_tensorboard(cls , * , open_browser : bool = False):
        """
        Interactively pick a packed ``.tar`` of TensorBoard logs, unpack to temp, and launch.
        """
        packed_tars = list(PATH.tsboard.glob('*.tar'))
        if len(packed_tars) == 0:
            Logger.alert1("No packed Tensorboard logs found in tar folder")
            return
        Logger.success("Choose from available packed Tensorboard tar files:")
        for i , packed_tar in enumerate(packed_tars):
            Logger.stdout(f"{i + 1:>2}. {packed_tar.name}" , indent = 1 , color = 'yellow')
        index = int(input("Enter the number of the packed Tensorboard tar to launch: "))
        assert index >= 1 and index <= len(packed_tars) , f"Invalid index: {index} , must be between 1 and {len(packed_tars)}"
        packed_tar = packed_tars[index - 1]
        log_dir = PATH.tsboard.joinpath('temp_run')
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(packed_tar, log_dir)
        cls.call_tensorboard(log_dir , open_browser = open_browser)

    @classmethod
    def run_trained_models_tensorboard(cls , * , open_browser : bool = False):
        """
        Interactively pick a trained model with preserved tensorboard snapshots and launch TB.
        """
        from src.api.model import ModelAPI
        from src.res.model.util import ModelPath
        candidates = [ModelPath(model) for model in ModelAPI.available_models(include_short_test = True)]
        candidates = [model for model in candidates if model.snapshot('tensorboard').exists()]
        if len(candidates) == 0:
            Logger.alert1("No available models with tensorboard logs found")
            return
        Logger.success("Choose from available models that have tensorboard logs when no model name is provided:")
        for i , model in enumerate(candidates):
            Logger.stdout(f"{i + 1:>2}. {model.full_name}" , indent = 1 , color = 'yellow')
        index = int(input("Enter the number of the model to launch: "))
        assert index >= 1 and index <= len(candidates) , f"Invalid index: {index} , must be between 1 and {len(candidates)}"
        model_name = candidates[index - 1].full_name
        log_dir = ModelPath(model_name).snapshot('tensorboard')
        
        cls.call_tensorboard(log_dir , open_browser = open_browser)

    @classmethod
    def launch(cls , * , open_browser : bool = False):
        """
        launch TensorBoard for local runs, packed archives, or trained-model logs.
        """
        os.chdir(PATH.main)
        Logger.success("Will launch TensorBoard for the following options:")

        options = ["local Tensorboard logs in run folder (lastest training)" , "Packed Tensorboard tar files (all past trainings)" , "Tensorboard logs of Trained Models (preserved training logs)"]
        for i , option in enumerate(options):
            Logger.stdout(f"{i + 1:>2}. {option}" , indent = 1 , color = 'yellow')
        index = int(input("Enter the number of the option to launch: "))
        assert index >= 1 and index <= len(options) , f"Invalid index: {index} , must be between 1 and {len(options)}"
        
        if index == 1:
            cls.run_local_tensorboard(open_browser = open_browser)
        elif index == 2:
            cls.run_packed_tensorboard(open_browser = open_browser)
        elif index == 3:
            cls.run_trained_models_tensorboard(open_browser = open_browser)

