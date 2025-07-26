from pathlib import Path
from typing import Literal
import json , re

BASE_DIR = Path(__file__).parent.parent
assert BASE_DIR.name == 'learndl' , f'BASE_DIR {BASE_DIR} not right , should be learndl'

RUNS_DIR = BASE_DIR.joinpath("src_runs")
CONF_DIR = BASE_DIR.joinpath("configs")

runs_db_dir = Path(__file__).with_name('.db')
runs_db_dir.mkdir(parents=True, exist_ok=True)

runs_db_path = runs_db_dir / 'task_manager.db'

log_dir = runs_db_dir / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

def st_log_file(log_type : Literal['action' , 'error']):
    file_path = log_dir / f'{log_type}.log'
    file_path.touch(exist_ok=True)
    return file_path

