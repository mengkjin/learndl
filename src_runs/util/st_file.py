from pathlib import Path
from typing import Literal
import json , re

BASE_DIR = Path(__file__).parent.parent
assert BASE_DIR.name == 'src_runs' , f'BASE_DIR {BASE_DIR} not right'

st_page_files_dir = BASE_DIR / 'st' / '_page_files'
st_page_files_dir.mkdir(parents=True, exist_ok=True)

st_log_dir = st_page_files_dir / 'logs'
st_log_dir.mkdir(parents=True, exist_ok=True)

st_exit_msg_dir = st_page_files_dir / 'exit_msg'
st_exit_msg_dir.mkdir(parents=True, exist_ok=True)

st_queue_dir = st_page_files_dir / 'queue'
st_queue_dir.mkdir(parents=True, exist_ok=True)

def queue_json_file(task_id : str):
    return st_queue_dir / f'{id_to_stem(task_id)}.json'

def id_to_stem(task_id : str):
    return re.sub(r"[/@*?]", "_", task_id)

def update_exit_message(task_id : str | None , **kwargs):
    if not task_id: 
        raise ValueError('task_id is required')
    if not kwargs:
        raise ValueError(f'No exit message to update for {task_id}')
    content = {task_id: {k: v for k, v in kwargs.items() if v is not None}}
    with open(exit_message_file(task_id), 'w') as f:
        json.dump(content, f, indent=2)

def exit_message_file(task_id : str):
    return st_exit_msg_dir / f'{id_to_stem(task_id)}.json'

def st_log_file(log_type : Literal['action' , 'error']):
    file_path = st_log_dir / f'{log_type}.log'
    file_path.touch(exist_ok=True)
    return file_path

