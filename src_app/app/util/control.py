import streamlit as st

from dataclasses import dataclass , field
from typing import Any
from pathlib import Path
from datetime import datetime
import time
import re

from src_app.db import RUNS_DIR
from src_app.backend import TaskQueue , TaskItem , TaskDatabase , ScriptRunner , PathItem
from src_app.frontend import YAMLFileEditorState , ActionLogger

PAGE_DIR = Path(__file__).parent.parent.joinpath('pages')
assert PAGE_DIR.exists() , f"Page directory {PAGE_DIR} does not exist"

def runs_page_url(script_key : str):
    """get runs page url"""
    return "pages/_" + re.sub(r'[/\\]', '_', script_key)

def runs_page_path(script_key : str):
    """get runs page path"""
    return PAGE_DIR.joinpath(runs_page_url(script_key).split('/')[-1])

@st.cache_resource
def get_cached_task_db() -> TaskDatabase:
    """get cached task database manager"""
    return TaskDatabase()

@dataclass
class SessionControl:
    """session control"""
    script_runners : dict[str, ScriptRunner] = field(default_factory=dict)
    current_script_runner : str | None = None
    current_task_item : str | None = None
    script_runner_trigger_item : str | None = None
    choose_task_item : str | None = None

    task_queue : TaskQueue | Any = None
    queue_last_action : tuple[str, bool] | None = None
    
    running_report_queue : str | None = None
    running_report_main : str | None = None
    running_report_main_cleared : bool = True

    running_report_file_previewer : Path | None = None

    script_params_cache : dict[str, dict[str, dict[str , Any]]] = field(default_factory=dict)
    config_editor_state : YAMLFileEditorState | None = None

    initialized : bool = False
    
    def __post_init__(self):
        self.task_db = get_cached_task_db()
        self.task_queue = TaskQueue('script_runner_main' , 100 , self.task_db)
        self.path_items = PathItem.iter_folder(RUNS_DIR, min_level = 0, max_level = 2)

        # delete all script detail files
        [page_path.unlink()  for page_path in PAGE_DIR.glob('_*.py')]

        # make script detail file
        [self.make_script_detail_file(item) for item in self.path_items if not item.is_dir]

    def __str__(self):
        return f"SessionControl()"
    
    @staticmethod
    def make_script_detail_file(item : PathItem):
        """make script detail file"""
        if item.is_dir: return
        with open(runs_page_path(item.script_key), 'w') as f:
            f.write(f"""
from util import starter , show_script_detail    

def main():
    starter()
    show_script_detail('{repr(item.script_key)}') 

if __name__ == '__main__':
    main()
""")

    def initialize(self):
        if 'session_control' not in st.session_state:
            st.session_state.session_control = self
        self.config_editor_state = YAMLFileEditorState.get_state('config_editor')
        self.initialized = True
        return self
    
    def filter_task_queue(self):
        """filter task queue"""
        assert self.initialized , 'SessionControl is not initialized'
        status_filter = st.session_state.get('queue-filter-status')
        folder_filter = st.session_state.get('queue-filter-path-folder')
        file_filter = st.session_state.get('queue-filter-path-file')
        filtered_queue = self.task_queue.filter(status = status_filter,
                                                folder = folder_filter,
                                                file = file_filter)
        return filtered_queue
    
    def latest_task_queue(self , num : int = 10):
        return self.task_queue.latest(num)

    def ready_to_go(self , obj : ScriptRunner):
        assert self.initialized , 'SessionControl is not initialized'
        return all(self.script_params_cache.get(obj.script_key, {}).get('valid', {}).values())
    
    def click_queue_item(self , item : TaskItem):
        """click queue item"""
        if self.running_report_queue is not None and self.running_report_queue == item.id:
            self.running_report_queue = None
        else:
            self.running_report_queue = item.id
        self.queue_last_action = None

    @ActionLogger.log_action()
    def click_queue_clear(self):
        """click task queue clear"""
        self.task_queue.clear()
        self.queue_last_action = f"Queue Clear Success" , True
    
    @ActionLogger.log_action()
    @st.dialog("Are You Sure about Clearing Queue?")
    def click_queue_clear_confirmation(self):
        """click task queue refresh confirmation"""
        col1 , col2 = st.columns(2 , gap = 'small')
        if col1.button("**Confirm**" , icon = ":material/check_circle:" , key = "confirm-clear-confirm-queue"):
            self.task_queue.clear()
            self.queue_last_action = f"Queue Clear Success" , True
            st.rerun()
        if col2.button("**Abort**" , icon = ":material/cancel:" , key = "confirm-clear-abort-queue"):
            self.queue_last_action = f"Queue Clear Aborted" , False
            st.rerun()

    def click_queue_filter_status(self):
        """click task queue filter status"""
        self.queue_last_action = f"Queue Filter Status: {st.session_state.get('queue-filter-status')}" , True

    def click_queue_filter_path_folder(self):
        """click task queue filter path folder"""
        folder_options = st.session_state.get('queue-filter-path-folder')
        if folder_options is not None:
            folder_options = [item.script_key for item in folder_options]
        self.queue_last_action = f"Queue Filter Path Folder: {folder_options}" , True
        
    def click_queue_filter_path_file(self):
        """click task queue filter path file"""
        file_options = st.session_state.get('queue-filter-path-file')
        if file_options is not None:
            file_options = [item.script_key for item in file_options]
        self.queue_last_action = f"Queue Filter Path File: {file_options}" , True

    @st.dialog("Are You Sure about Clearing Logs?")
    def click_log_clear_confirmation(self):
        """click log clear confirmation"""
        col1 , col2 = st.columns(2 , gap = 'small')
        if col1.button("**Confirm**" , icon = ":material/check_circle:" , key = "confirm-clear-confirm-log"):
            ActionLogger.clear_log()
            self.queue_last_action = f"Log Clear Success" , True
            st.rerun()
        if col2.button("**Abort**" , icon = ":material/cancel:" , key = "confirm-clear-abort-log"):
            self.queue_last_action = f"Log Clear Aborted" , False
            st.rerun()

    @ActionLogger.log_action()
    def click_queue_sync(self):
        """click task queue sync"""
        self.task_queue.sync()
        self.queue_last_action = f"Queue Manually Synced at {datetime.now().strftime('%H:%M:%S')}" , True
   
    @ActionLogger.log_action()
    def click_queue_refresh(self):
        """click task queue refresh"""
        self.task_queue.refresh()
        self.queue_last_action = f"Queue Manually Refreshed at {datetime.now().strftime('%H:%M:%S')}" , True

    @ActionLogger.log_action()
    def click_queue_empty(self):
        """click task queue empty"""
        self.task_queue.empty()
        self.queue_last_action = f"Queue Emptied" , True

    @ActionLogger.log_action()
    def click_queue_remove_item(self , item : TaskItem):
        """click task queue remove item"""
        if item.kill():
            self.queue_last_action = f"Remove Success: {item.id}" , True
        else:
            self.queue_last_action = f"Remove Failed: {item.id}" , False
        self.task_queue.remove(item)

    @ActionLogger.log_action()
    def click_script_runner_expand(self , runner : ScriptRunner):
        """click script runner expand"""
        if self.current_script_runner is not None and self.current_script_runner == runner.script_key:
            self.current_script_runner = None
        else:
            self.current_script_runner = runner.script_key
        
    def click_script_runner_filter(self , runner : ScriptRunner):
        """click script runner filter"""
        st.session_state['queue-filter-status'] = 'All'
        st.session_state['queue-filter-path-folder'] = []
        st.session_state['queue-filter-path-file'] = [runner.path.path]
        
    @ActionLogger.log_action()
    def click_script_runner_run(self , runner : ScriptRunner , params : dict[str, Any]):
        """click run button"""
        run_params = {
            'email': int(runner.header.email),
            'close_after_run': bool(runner.header.close_after_run)
        }
        run_params.update(params)
        item = runner.run_script(self.task_queue , **run_params)
        self.current_task_item = item.id
        self.queue_last_action = f"Add to Queue: {item.id}" , True
        
        if self.running_report_main != item.id:
            self.running_report_main = item.id
            self.running_report_main_cleared = False
        

    @ActionLogger.log_action()
    def click_file_preview(self , path : Path):
        """click file previewer"""
        if self.running_report_file_previewer == path:
            self.running_report_file_previewer = None
        else:   
            self.running_report_file_previewer = path   

    @ActionLogger.log_action()
    def click_file_download(self , path : Path):
        """click file previewer"""
        # TODO: things to do before download

    @ActionLogger.log_action()
    def click_show_complete_report(self , item : TaskItem):
        """click show complete report"""
        self.current_script_runner = item.runner_script_key
        self.current_task_item = item.id
        self.script_runner_trigger_item = item.id
        self.running_report_main = item.id
        self.running_report_main_cleared = False
        self.running_report_file_previewer = None

    @ActionLogger.log_action()
    def click_item_choose_select(self , item : TaskItem):
        """click choose task item"""
        self.current_script_runner = item.runner_script_key
        new_id = item.id if self.choose_task_item != item.id else None
        
        self.current_task_item = new_id
        self.script_runner_trigger_item = new_id
        self.running_report_main = new_id
        self.choose_task_item = new_id
        self.running_report_main_cleared = False
        self.running_report_file_previewer = None

    @ActionLogger.log_action()
    def click_choose_item_selectbox(self):
        """click choose task item"""
        item_id = st.session_state['choose-item-selectbox']
        item = self.task_queue.get(item_id)
        if item is None:
            return
        
        self.current_script_runner = item.runner_script_key
        new_id = item.id if self.choose_task_item != item.id else None
        
        self.current_task_item = new_id
        self.script_runner_trigger_item = new_id
        self.running_report_main = new_id
        self.choose_task_item = new_id
        self.running_report_main_cleared = False
        self.running_report_file_previewer = None

    @ActionLogger.log_action()
    def click_item_choose_remove(self , item : TaskItem):
        """click remove task item"""
        self.task_queue.remove(item)
        self.running_report_main = None
        self.running_report_main_cleared = False
        self.running_report_file_previewer = None
    
    @staticmethod
    def wait_for_complete(item : TaskItem , running_timeout : int = 20):
        """wait for complete"""
        while True:
            item.refresh()
            if item.status in ['complete' , 'error']:
                return True
            if item.status == 'starting':
                running_timeout -= 1
            if running_timeout <= 0:
                raise RuntimeError(f'Script {item.script} running timeout! Still starting')
            time.sleep(1)
        return False
   
SC = SessionControl().initialize()