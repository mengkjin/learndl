import subprocess , os
import numpy as np

from datetime import datetime
from pathlib import Path

from src.proj import MACHINE , Logger , PATH , SharedSync

_share_folder = MACHINE.share_folder_path()
if _share_folder is not None:
    root_path = _share_folder.joinpath('task_schedule')
    root_path.mkdir(parents=True, exist_ok=True)
else:
    root_path = Path()

class ScheduledTask:
    """a task that is scheduled to run at a specific time"""
    _instances : dict[str , dict[str , 'ScheduledTask']] = {}

    def __init__(self, operator_machine : str, cmdline : str, distribution_time : int, time_index : int):
        self.operator_machine = operator_machine
        self.cmdline = cmdline
        self.distribution_time = distribution_time
        self.time_index = time_index

        assert self.time_index < 100 , f'time_index is too large: {self.time_index}'

        if self.operator_machine not in self._instances:
            self._instances[self.operator_machine] = {}

        if self.time_key not in self._instances[self.operator_machine]:
            self._instances[self.operator_machine][self.time_key] = self
        else:
            raise ValueError(f'machine {self.operator_machine} time_key {self.time_key} already exists')

    def __repr__(self):
        return f'ScheduledTask(operator_machine={self.operator_machine},cmdline={self.cmdline},distribution_time={self.distribution_time},time_index={self.time_index})'

    @classmethod
    def new_task(cls, operator_machine : str, cmdline : str):
        """add new task"""
        distribution_time = int(datetime.now().strftime('%Y%m%d%H%M%S'))
        existing_time_index = np.array([int(key.split('.')[1]) for key in cls._instances.get(operator_machine , {}).keys() if key.startswith(f'{distribution_time}.')])
        time_index = np.setdiff1d(np.arange(200), existing_time_index).min()
        cls.check_cmdline_exists(operator_machine, cmdline)
        return cls(operator_machine, cmdline, distribution_time, time_index).distribute()

    @classmethod
    def load_task(cls, file_path : Path):
        """get the task from the file path"""
        assert file_path.exists() , f'{file_path} does not exist'
        assert file_path.is_file() , f'{file_path} is not a file'
        assert file_path.suffix == '.await' , f'{file_path} is not a await file'
        operator_machine = file_path.parent.name
        cmdline = file_path.read_text()
        distribution_time = int(file_path.stem.split('.')[0])
        time_index = int(file_path.stem.split('.')[1])
        return cls(operator_machine, cmdline, distribution_time, time_index)
    
    @classmethod
    def check_cmdline_exists(cls, operator_machine : str, cmdline : str):
        """check if the cmdline exists in the task"""
        existing_cmdlines = [task.cmdline for task in list(cls._instances.get(operator_machine , {}).values())]
        if cmdline in existing_cmdlines:
            Logger.error(f'!!!Cmdline already exists in {operator_machine}')
            Logger.error(f'!!!Cmdline: {cmdline}')

    def distribute(self):
        """distribute the task"""
        if not root_path:
            return self
        self.task_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.task_path, 'w') as f:
            f.write(self.cmdline)
        Logger.info(f"ScheduledTask {self.task_path} distributed")
        return self

    def complete(self) -> None:
        """complete the task"""
        self.task_path.rename(self.task_path.with_suffix('.done'))
        del self._instances[self.operator_machine][self.time_key]

    @property
    def task_path(self) -> Path:
        """return the path of the task (with .await suffix)"""
        if not root_path:
            return Path()
        return root_path.joinpath(self.operator_machine).joinpath(f'{self.time_key}.await')

    @property
    def time_key(self) -> str:
        """return the time key : distribution_time.time_index of the task"""
        return f'{self.distribution_time}.{self.time_index:02d}'

    def run(self):
        """run the task"""
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(MACHINE.main_path)
            # execute the command
            result = subprocess.run(
                self.cmdline, 
                shell=True,
                encoding='utf-8',
                env = env,
                check = True,
            )
            # printout the result
            Logger.info(f"command {self.cmdline} executed successfully, return code: {result.returncode}\n")
            self.complete()
        except subprocess.CalledProcessError as e:
            Logger.error(f"command executed failed, return code: {e.returncode}")
            Logger.error(f"error output: {e.stderr}")
            Logger.error("stop executing subsequent commands\n")
        except Exception as e:
            Logger.error(f"command executed failed, exception: {e}")
            Logger.error("stop executing subsequent commands\n")
        return self

    @classmethod
    def get_all_tasks(cls):
        """get all the tasks"""
        cls._instances.clear()
        [cls.load_task(file) for file in root_path.rglob('*.await')]
        return cls._instances 

class TaskScheduler:
    machine_name = 'mengkjin-server'
    def __bool__(self):
        return _share_folder is not None

    @staticmethod
    def get_machine_tasks() -> dict[str , 'ScheduledTask']:
        """get all the tasks"""
        SharedSync.sync()
        return ScheduledTask.get_all_tasks().get(MACHINE.name , {})

    @classmethod
    def add_task(cls, machine_name : str = 'mengkjin-server', cmdline : str = ''):
        if not cmdline:
            return
        cls.get_machine_tasks()
        return ScheduledTask.new_task(machine_name, cmdline)

    @classmethod
    def print_machine_tasks(cls):
        """printout all the tasks of the machine"""
        tasks = cls.get_machine_tasks()
        for task in tasks.values():
            Logger.warning(f'{task} awaits to be executed')
        return tasks

    @classmethod
    def run_machine_tasks(cls):
        """run all the tasks of the machine"""
        SharedSync.sync()
        tasks = cls.get_machine_tasks()
        for task in tasks.values():
            task.run()

    @classmethod
    def add_run_script(cls, script : str | Path , kwargs : dict | None = None , machine_name : str = 'mengkjin-server'):
        """add a script to run"""
        kwargs = kwargs or {}
        if isinstance(script , str):
            script = PATH.scpt.joinpath(script).with_suffix('.py')
            assert Path(script).exists() , f'script {script} does not exist'
        else:
            script = str(script.absolute())
        script = PATH.path_at_machine(script , machine_name)
        kwargs_str = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in kwargs.items() if v != ''])
        cmdline = f"{PATH.path_at_machine(Path(MACHINE.python_path) , machine_name)} {script} {kwargs_str}"
        Logger.info(f"add run script {script} with kwargs {kwargs} to {machine_name}")
        return cls.add_task(machine_name = machine_name, cmdline = cmdline)