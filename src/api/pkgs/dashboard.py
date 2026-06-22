"""
API for launching dashboard of this project.
"""

from __future__ import annotations
import os , shutil , subprocess , webbrowser , time , socket
from pathlib import Path
from src.proj import PATH , Logger , Base
from src.proj.util.functional.ask import AskFor

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
        db_path = f'sqlite:///{db_path.resolve().as_posix()}'
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
    def run_latest_optuna_dashboard(cls , * , open_browser : bool = False , **kwargs):
        """
        Start Optuna Dashboard pointing at ``PATH.optuna/run`` if logs exist.
        """
        candidates = list(PATH.optuna.rglob('*.sqlite3'))
        if not candidates:
            Logger.alert1("No local Optuna Dashboard database found in optuna folder")
            return
        candidates.sort(key = lambda x: x.stat().st_mtime , reverse = True)
        cls.call_optuna_dashboard(candidates[0] , open_browser = open_browser)

    @classmethod
    def run_saved_optuna_dashboard(cls , * , open_browser : bool = False , **kwargs):
        """
        Interactively pick a saved Optuna Dashboard database and launch it.
        """
        candidates = list(PATH.optuna.rglob('*.sqlite3'))
        if not candidates:
            Logger.alert1("No available Optuna Dashboard databases found")
            return
        flag = AskFor.Selections([
            cand.relative_to(PATH.optuna) for cand in candidates] , 
            confirm=False , multiple=False , 
            title = f'Choose from available Optuna Dashboard databases:')
        if flag.result is None:
            return
        cls.call_optuna_dashboard(candidates[flag.result - 1] , open_browser = open_browser)

    @classmethod
    def choose_to_delete_optuna_record(cls , **kwargs):
        """
        Interactively pick Optuna Dashboard database(s) to delete.
        """
        for loop in AskFor.LoopTillExit(message = f'Do you want to delete more Optuna Dashboard databases?'):
            candidates = list(PATH.optuna.rglob('*.sqlite3'))
            if not candidates:
                Logger.alert1("No available Optuna Dashboard databases found")
                return
            flag = AskFor.Options(
                [db_path.relative_to(PATH.optuna) for db_path in candidates] , 
                multiple=True , confirm=True ,
                title = f'Choose from available Optuna Dashboard databases to delete:')
            if loop.set_flag(flag):
                for db_path in flag.results:
                    PATH.optuna.joinpath(db_path).resolve().unlink()
                Logger.success(f"Deleted {len(flag.results)} Optuna Dashboard databases")

    @classmethod
    def launch_option_menu(cls , * , open_browser : bool = False):
        """
        launch Optuna Dashboard CLI menu for local runs, saved databases.
        """
        os.chdir(PATH.main)
        Logger.success("What do you want to do with Optuna Dashboard?")
        options = {
            "Latest Run Optuna Dashboard database" : cls.run_latest_optuna_dashboard ,
            "Select Saved Optuna Dashboard database" : cls.run_saved_optuna_dashboard ,
            "Clear Optuna Dashboard database" : cls.choose_to_delete_optuna_record ,
        }
        flag = AskFor.Options(
            list(options.keys()) , multiple=False , confirm=False , 
            title = f'What do you want to do with Optuna Dashboard?'
        )
        if flag.result:
            options[flag.result](open_browser = open_browser)
        return flag

    @classmethod
    def launch(cls , * , open_browser : bool = False):
        """
        launch Optuna Dashboard menu iteratively.
        """
        for loop in AskFor.LoopTillExit(False):
            loop.set_flag(cls.launch_option_menu(open_browser = open_browser))

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
    def run_local_tensorboard(cls , * , open_browser : bool = False , **kwargs):
        """
        Start TensorBoard pointing at ``PATH.tensorboard/run`` if logs exist.
        """
        log_dir = PATH.tsboard.joinpath('run')
        if not log_dir.exists():
            Logger.alert1("No local Tensorboard logs found in run folder")
            return
        cls.call_tensorboard(log_dir , open_browser = open_browser)

    @classmethod
    def run_trained_models_tensorboard(cls , * , open_browser : bool = False    ):
        """
        Interactively pick a trained model with preserved tensorboard snapshots and launch TB.
        """
        from src.api.pkgs.model import ModelAPI
        from src.res.model.util import ModelPath
        def predicate(model : ModelPath) -> bool:
            """predicate to check if a model has tensorboard logs and at least one subfolder with more than one file"""
            if not model.snapshot('tensorboard').is_dir():
                return False
            for sub in model.snapshot('tensorboard').iterdir():
                if sub.is_dir():
                    if len(list(sub.glob('*'))) > 1:
                        return True
            return False
        candidates = [ModelPath(model) for model in ModelAPI.available_models(include_short_test = True)]
        candidates = [model for model in candidates if predicate(model)]
        if len(candidates) == 0:
            Logger.alert1("No available models with tensorboard logs found")
            return
        flag = AskFor.Options(
            [model.full_name for model in candidates] , 
            multiple=False , confirm=False ,
            title = f'Which model to launch TensorBoard?')
        if flag.result is None:
            return
        cls.call_tensorboard(ModelPath(flag.result).snapshot('tensorboard') , open_browser = open_browser)

    @classmethod
    def run_packed_tensorboard(cls , * , open_browser : bool = False , **kwargs):
        """
        Interactively pick a packed ``.tar`` of TensorBoard logs, unpack to temp, and launch.
        """
        packed_tars = list(PATH.tsboard.glob('*.tar'))
        if len(packed_tars) == 0:
            Logger.alert1("No packed Tensorboard logs found in tar folder")
            return
        flag = AskFor.Options(
            [packed_tar.name for packed_tar in packed_tars] , 
            multiple=False , confirm=False ,
            title = f'Choose from available packed Tensorboard tar files:')
        if flag.result is None:
            return

        log_dir = PATH.tsboard.joinpath('temp_run')
        shutil.rmtree(log_dir , ignore_errors=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(PATH.tsboard.joinpath(flag.result), log_dir)
        cls.call_tensorboard(log_dir , open_browser = open_browser)

    @classmethod
    def choose_to_delete_tensorboard_record(cls , **kwargs):
        """
        Interactively pick TensorBoard log(s) to delete.
        """
        for loop in AskFor.LoopTillExit(message = f'Do you want to delete more TensorBoard logs?'):
            candidates = list(PATH.tsboard.glob('*.tar'))
            if not candidates:
                Logger.alert1("No available TensorBoard logs found")
                return
            flag = AskFor.Options(
                [log_dir.relative_to(PATH.tsboard) for log_dir in candidates] , 
                multiple=True , confirm=True ,
                title = f'Choose from available TensorBoard logs to delete:')
            if loop.set_flag(flag):
                for log_dir in flag.results:
                    PATH.tsboard.joinpath(log_dir).resolve().unlink()
                Logger.success(f"Deleted {len(flag.results)} TensorBoard logs")

    @classmethod
    def launch_option_menu(cls , * , open_browser : bool = False):
        """
        launch TensorBoard CLI menu for local runs, packed archives, or trained-model logs.
        """
        os.chdir(PATH.main)
        Logger.success("Will launch TensorBoard for the following options:")

        options = {
            "Latest Tensorboard logs in run folder" : cls.run_local_tensorboard ,
            "Trained Models Tensorboard logs" : cls.run_trained_models_tensorboard ,
            "Packed Tensorboard tar files (all past trainings)" : cls.run_packed_tensorboard ,
            "Delete TensorBoard logs" : cls.choose_to_delete_tensorboard_record ,
        }
        flag = AskFor.Options(list(options.keys()) , multiple=False , confirm=False , title = f'What do you want to do with TensorBoard?')
        if flag.result:
            options[flag.result](open_browser = open_browser)
        return flag

    @classmethod
    def launch(cls , * , open_browser : bool = False):
        """
        launch TensorBoard menu iteratively.
        """
        for loop in AskFor.LoopTillExit(False):
            loop.set_flag(cls.launch_option_menu(open_browser = open_browser))