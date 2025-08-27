from pathlib import Path
from typing import Literal
import socket

BASE_DIR = Path(__file__).parent.parent.parent
assert BASE_DIR.name == 'learndl' , f'BASE_DIR {BASE_DIR} not right , should be learndl'

SCPT_DIR = BASE_DIR.joinpath("src" , "scripts")
CONF_DIR = BASE_DIR.joinpath("configs")

_machine_name = socket.gethostname().split('.')[0]

_db_dir = BASE_DIR / '.local_resources' / 'app' / _machine_name
_db_dir.mkdir(parents=True, exist_ok=True)

def get_task_db_path():
    return _db_dir / 'task_manager.db'

def get_st_log_path(log_type : Literal['action' , 'error']):
    file_path = _db_dir / f'page_{log_type}.log'
    file_path.touch(exist_ok=True)
    return file_path

