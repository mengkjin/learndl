import streamlit as st

from dataclasses import dataclass , field
from typing import Any , ClassVar , Callable
from pathlib import Path
from datetime import datetime

from src.proj import PATH , Logger
from src.interactive.backend import TaskQueue , TaskItem , TaskDatabase , ScriptRunner , PathItem
from src.interactive.frontend import YAMLFileEditorState , action_confirmation , ParamCache # , ActionLogger

def set_current_page(key: str) -> None:
    st.session_state["current_page"] = key

@st.cache_resource
def get_cached_task_db() -> TaskDatabase:
    """get cached task database manager"""
    return TaskDatabase()

def universal_action(func : Callable):
    def wrapper(*args , **kwargs):
        ret = func(*args , **kwargs)
        if SessionControl._instance is not None:
            SessionControl._instance.task_queue.refresh()
        return ret
    return wrapper

@dataclass
class SessionControl:
    """session control"""
    # universal
    script_runners : dict[str, ScriptRunner] = field(default_factory=dict)
    task_queue : TaskQueue | Any = None
    current_task_item : str | None = None

    # for task queue page
    queue_last_action : tuple[str, bool] | None = None
    running_report_queue : str | None = None

    # for task detail page
    running_report_init : bool = True
    running_report_file_previewer : Path | None = None
    param_inputs_form : Any = None
    script_params_cache : ParamCache = field(default_factory=ParamCache)

    # for config editor page
    config_editor_state : YAMLFileEditorState | None = None

    _instance : 'ClassVar[SessionControl | None]' = None
    
    def __post_init__(self):
        self.__class__._instance = self
        self.task_db = get_cached_task_db()
        self.task_queue = TaskQueue(task_db = self.task_db)
        self.path_items = PathItem.iter_folder(PATH.scpt, min_level = 0, max_level = 2)
        self.task_queue.refresh()
        if 'session_control' not in st.session_state:
            st.session_state.session_control = self
        self.config_editor_state = YAMLFileEditorState.get_state('config_editor')

    def __str__(self):
        return f"SessionControl()"

    @universal_action
    def switch_page(self , page_name : str):
        self.current_page_name = page_name 
    
    def get_script_runner(self , script_key : str) -> ScriptRunner:
        if script_key not in self.script_runners: 
            self.script_runners[script_key] = ScriptRunner.from_key(script_key)
        runner = self.script_runners[script_key]
        if runner is None:
            raise ValueError(f"Script {script_key} not found in SC.script_runners")
        return runner
    
    def get_task_item(self , task_id : str | None = None) -> TaskItem | None:
        if self.task_queue is None or task_id is None:
            return None
        return self.task_queue.get(task_id)
    
    def get_latest_task_item(self , script_key : str | None = None) -> TaskItem | None:
        return self.task_queue.latest(script_key)
        
    def clear_report_placeholder(self):
        if 'task_report_placeholder' in st.session_state:
            with st.session_state['task_report_placeholder']:
                st.write('')

    def call_report_placeholder(self):
        placeholder = st.empty()
        if 'task_report_placeholder' not in st.session_state:
            st.session_state['task_report_placeholder'] = placeholder
        return placeholder
    
    def get_filtered_queue(self):
        """filter task queue"""
        status_filter = st.session_state.get('task-filter-status')
        source_filter = st.session_state.get('task-filter-source')
        folder_filter = st.session_state.get('task-filter-path-folder')
        file_filter = st.session_state.get('task-filter-path-file')
        filtered_queue = self.task_queue.filter(status = status_filter,
                                                source = source_filter,
                                                folder = folder_filter,
                                                file = file_filter)
        return filtered_queue
    
    def get_latest_queue(self , num : int = 10):
        return self.task_queue.latest_n(num)
    
    def get_global_settings(self):
        setting = {}
        max_vb = str(st.session_state.get('global-settings-maxvb' , 'none')).lower()
        if max_vb != 'none': 
            setting['max_vb'] = 1 if max_vb.startswith('y') else 0

        email = str(st.session_state.get('global-settings-email' , 'none')).lower()
        if email != 'none': 
            setting['email'] = 1 if email.startswith('y') else 0

        mode = str(st.session_state.get('global-settings-mode' , 'none')).lower()
        if mode != 'none': 
            setting['mode'] = 'shell' if mode == 'shell' else 'os'
        
        return setting

    def get_script_runner_cmd(self , runner : ScriptRunner | None , params : dict[str, Any] | None , 
                              operation_txt = True):
        """preview runner cmd"""
        if runner is None: 
            return None
        run_params = self.get_global_settings()
        if 'email' not in run_params: 
            run_params['email'] = runner.header.email
        if 'mode'  not in run_params: 
            run_params['mode']  = runner.header.mode
        if params: 
            run_params.update(params)
        cmd = runner.preview_cmd(**run_params)
        if operation_txt:
            run_text = 'run' if 'mode' not in run_params else f'{run_params["mode"]} run'
            cmd = f":blue[**{run_text.title()}**]: {cmd}"
        return cmd
    
    def get_script_runner_validity(self , params : dict[str, Any] | None):
        params = params or {}
        for pname , pvalue in self.param_inputs_form.param_dict.items():
            if pvalue.required and params.get(pname) is None:
                return False
        return True
   
    @universal_action
    def click_queue_item(self , item : TaskItem):
        """click queue item"""
        if self.running_report_queue is not None and self.running_report_queue == item.id:
            self.running_report_queue = None
        else:
            self.running_report_queue = item.id
        self.queue_last_action = None
    
    def change_queue_filter_status(self):
        """click task queue filter status"""
        self.queue_last_action = f"Task Filter Status: {st.session_state.get('task-filter-status')}" , True

    def change_queue_filter_source(self):
        """click task queue filter source"""
        self.queue_last_action = f"Task Filter Source: {st.session_state.get('task-filter-source')}" , True

    def change_queue_filter_path_folder(self):
        """click task queue filter path folder"""
        folder_options = st.session_state.get('task-filter-path-folder')
        if folder_options is not None:
            folder_options = [item.name for item in folder_options]
        self.queue_last_action = f"Task Filter Path Folder: {folder_options}" , True
        
    def change_queue_filter_path_file(self):
        """click task queue filter path file"""
        file_options = st.session_state.get('task-filter-path-file')
        if file_options is not None:
            file_options = [item.name for item in file_options]
        self.queue_last_action = f"Task Filter Path File: {file_options}" , True

    @universal_action
    def click_log_clear_confirmation(self):
        """click log clear confirmation"""
        def on_confirm():
            # ActionLogger.clear_log()
            self.queue_last_action = f"Log Clear Success" , True
        def on_abort():
            self.queue_last_action = f"Log Clear Aborted" , False
        action_confirmation(on_confirm , on_abort , title = "Are You Sure about Clearing Logs (This Action is Irreversible)?")

    # @ActionLogger.log_action()
    @universal_action
    def click_queue_sync(self):
        """click task queue sync"""
        self.task_queue.sync()
        self.queue_last_action = f"Queue Manually Synced at {datetime.now().strftime('%H:%M:%S')}" , True
   
    # @ActionLogger.log_action()
    @universal_action
    def click_queue_refresh(self):
        """click task queue refresh"""
        self.task_queue.refresh()
        self.queue_last_action = f"Queue Manually Refreshed at {datetime.now().strftime('%H:%M:%S')}" , True

    # @ActionLogger.log_action()
    @universal_action
    def click_queue_clean(self):
        """click task queue refresh confirmation"""
        items = [item for item in self.task_queue.values() if item.is_error or item.is_killed]
        [self.task_queue.remove(item) for item in items]
        self.queue_last_action = f"Queue Cleaned All Error Tasks at {datetime.now().strftime('%H:%M:%S')}" , True

    # @ActionLogger.log_action()
    @universal_action
    def click_queue_delist_all(self):
        """click task queue delist all"""
        self.task_queue.empty()
        self.queue_last_action = f"Entire Queue Delisted" , True

    # @ActionLogger.log_action()
    @universal_action
    def click_queue_remove_all(self):
        """click task queue refresh confirmation"""
        def on_confirm():
            self.task_queue.clear()
            self.queue_last_action = f"Entire Queue Removed Success" , True
        def on_abort():
            self.queue_last_action = f"Entire Queue Removal Aborted" , False
        action_confirmation(on_confirm , on_abort , 
                            title = "Are You Sure about Removing All Tasks in Queue (Will be Auto Backuped)?")

    # @ActionLogger.log_action()
    @universal_action
    def click_queue_delist_item(self , item : TaskItem):
        """click task queue delist item"""
        self.task_queue.delist(item)
        self.queue_last_action = f"Delist Success: {item.id}" , True

    # @ActionLogger.log_action()
    @universal_action
    def click_queue_remove_item(self , item : TaskItem):
        """click task queue remove item"""
        def on_confirm():
            if item.kill():
                self.queue_last_action = f"Remove Success: {item.id}" , True
            else:
                self.queue_last_action = f"Remove Failed: {item.id}" , False
            self.task_queue.remove(item , force = True)
        def on_abort():
            self.queue_last_action = f"Remove Aborted: {item.id}" , False

        if item.status != 'error':
            action_confirmation(on_confirm , on_abort , title = f"Are You Sure about Removing {item.id} (This Action is Irreversible)?")
        else:
            on_confirm()

    # @ActionLogger.log_action()
    @universal_action
    @st.dialog("Are You Sure about Restoring from Backup?")
    def click_queue_restore_all(self):
        """click task queue restore all"""
        available_backups = self.task_db.get_backup_paths()
        if len(available_backups) == 0:
            st.error("No backup files found")
            return
        backup_stats = {
            backup: self.task_db.backup_stats(backup)
            for backup in available_backups
        }
        
        backup_selectbox = st.selectbox(
            "Select Backup to Restore", available_backups, 
            format_func = lambda x: f"{x.name} ({backup_stats[x]['task_count']} Tasks, {backup_stats[x]['queue_count']} Queue)")
        col1 , col2 = st.columns(2 , gap = "small")
        if col1.button("Restore" , key = "task-queue-restore-all-confirm" , icon = ":material/restore:" , type = "primary"):
            self.task_db.restore_backup(backup_selectbox)
            self.task_queue.sync()
            self.queue_last_action = f"Restore Success" , True
            st.rerun()
        if col2.button("Abort" , key = "task-queue-restore-all-abort" , icon = ":material/cancel:" , type = "secondary"):
            self.queue_last_action = f"Restore Aborted" , False
            st.rerun()
        
    @universal_action
    def click_script_runner_filter(self , runner : ScriptRunner):
        """click script runner filter"""
        st.session_state['task-filter-status'] = 'All'
        st.session_state['task-filter-source'] = 'All'
        st.session_state['task-filter-path-folder'] = []
        st.session_state['task-filter-path-file'] = [runner.path.path]
        
    # @ActionLogger.log_action()
    @universal_action
    def click_script_runner_run(self , runner : ScriptRunner , params : dict[str, Any] | None):
        """click run button"""
        run_params = self.get_global_settings()
        if 'email' not in run_params: 
            run_params['email'] = runner.header.email
        if 'mode'  not in run_params: 
            run_params['mode']  = runner.header.mode
        if params: 
            run_params.update(params)
        
        item = runner.build_task(self.task_queue , **run_params)

        self.queue_last_action = f"Add to Queue: {item.id}" , True
        self.current_task_item = item.id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

        item.run_script()

    # @ActionLogger.log_action()
    def click_file_preview(self , path : Path):
        """click file previewer"""
        Logger.stdout('click file previewer')
        if self.running_report_file_previewer == path:
            self.running_report_file_previewer = None
        else:   
            self.running_report_file_previewer = path   

    # @ActionLogger.log_action()
    def click_file_download(self , path : Path):
        """click file previewer"""
        # TODO: things to do before download

    # @ActionLogger.log_action()
    @universal_action
    def click_show_complete_report(self , item : TaskItem):
        """click show complete report"""
        self.current_task_item = item.id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

    # @ActionLogger.log_action()
    @universal_action
    def click_item_choose_select(self , item : TaskItem):
        """click choose task item"""
        new_id = item.id if self.current_task_item != item.id else None
        
        self.current_task_item = new_id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

    # @ActionLogger.log_action()
    @universal_action
    def click_choose_item_selectbox(self , item_id : str | None = None):
        """click choose task item"""
        if item_id is None: 
            item_id = st.session_state['choose-item-selectbox']
        item = self.task_queue.get(item_id)
        if item is None:
            return
        
        new_id = item.id if self.current_task_item != item.id else None
        
        self.current_task_item = new_id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None
        # st.session_state['script-task-selector-placeholder'].write('')
    
    @staticmethod
    def wait_until_completion(item : TaskItem , starting_timeout : int = 20):
        """wait for complete"""
        return item.wait_until_completion(starting_timeout)
   
SC = SessionControl()