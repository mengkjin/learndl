"""Session state, action callbacks, and the control panel for the interactive app.

Key objects:

* :class:`SessionControl` (``SC``) — dataclass-based singleton that owns the
  task queue, script runners, param cache, and all button click handlers.
  Instantiated once at module import as the module-level ``SC``.
* :func:`universal_action` — decorator that refreshes the task queue after
  every callback so the UI always reflects the latest state.
* :class:`ControlPanelButton` / :class:`ControlPanel` — the shared action
  bar rendered at the top of every page via :meth:`SessionControl.get_control_panel`.
"""
import streamlit as st
import subprocess

from abc import abstractmethod , ABC
from dataclasses import dataclass , field
from typing import Any , ClassVar , Callable
from pathlib import Path
from datetime import datetime

from src.proj import PATH , Logger , Proj , MACHINE
from src.proj.util import Options
from src.interactive.backend import TaskQueue , TaskItem , TaskDatabase , ScriptRunner , PathItem
from src.interactive.frontend import YAMLFileEditorState , action_confirmation , ParamCache

def set_current_page(key: str) -> None:
    """Store ``key`` as the active page name in Streamlit session state."""
    st.session_state["current_page"] = key

@st.cache_resource
def get_cached_task_db() -> TaskDatabase:
    """get cached task database manager"""
    return TaskDatabase()

def universal_action(func : Callable) -> Callable:
    """Decorator: refresh the task queue after every UI callback.

    Wraps ``func`` so that :meth:`TaskQueue.refresh` is called on the active
    :class:`SessionControl` instance after each invocation, keeping the
    displayed queue in sync with the underlying database.
    """
    def wrapper(*args , **kwargs):
        ret = func(*args , **kwargs)
        if SessionControl._instance is not None:
            SessionControl._instance.task_queue.refresh()
        return ret
    return wrapper

@dataclass
class SessionControl:
    """Per-session singleton that wires together the backend and the UI.

    Owns the :class:`TaskDatabase`, :class:`TaskQueue`, per-script
    :class:`ScriptRunner` cache, parameter cache, and all Streamlit button
    click handlers.  The module-level ``SC`` instance is created at import
    time and is shared across every page in the same Streamlit session.

    Attributes:
        script_runners: Cache of :class:`ScriptRunner` objects keyed by script key.
        task_queue: The live task queue bound to the shared task database.
        current_task_item: ID of the task currently shown in the detail panel.
        queue_last_action: ``(message, success)`` tuple describing the last
            queue action, used to show success/error banners.
        running_report_queue: ID of the task whose inline report is expanded
            in the task-queue page.
        running_report_init: ``True`` on first render of a report so callers
            can trigger auto-scroll or other one-shot setup.
        running_report_file_previewer: Path of the exit file currently
            previewed, or ``None``.
        param_inputs_form: The active :class:`ParamInputsForm` for the current
            script page.
        script_params_cache: :class:`ParamCache` for persisting widget values
            across reruns.
        config_editor_state: :class:`YAMLFileEditorState` for the config editor
            page.
    """
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
    
    def __post_init__(self) -> None:
        """Initialize the instance and register it as the active singleton."""
        self.rerun()

    def __str__(self) -> str:
        return f"SessionControl()"

    def rerun(self) -> None:
        """Re-initialize mutable state: register singleton, reload queue and path items.

        Called in ``__post_init__`` and by :class:`ControlRefreshInteractiveButton`
        when the user explicitly refreshes the app.
        """
        self.__class__._instance = self
        self.task_db = get_cached_task_db()
        self.task_queue = TaskQueue(task_db = self.task_db)
        self.path_items = PathItem.iter_folder(PATH.scpt, min_level = 0, max_level = 2)
        self.task_queue.refresh()
        if 'session_control' not in st.session_state:
            st.session_state.session_control = self
        self.config_editor_state = YAMLFileEditorState.get_state('config_editor')

    @universal_action
    def switch_page(self , page_name : str) -> None:
        """Record the active page name and refresh the queue (via ``universal_action``)."""
        self.current_page_name = page_name
    
    def get_script_runner(self , script_key : str) -> ScriptRunner:
        """Return the :class:`ScriptRunner` for *script_key*, creating it on first access.

        Args:
            script_key: Relative script path, e.g. ``'4_train/1_train_model.py'``.

        Returns:
            The cached :class:`ScriptRunner` instance.

        Raises:
            ValueError: If the runner cannot be found.
        """
        if script_key not in self.script_runners:
            self.script_runners[script_key] = ScriptRunner.from_key(script_key)
        runner = self.script_runners[script_key]
        if runner is None:
            raise ValueError(f"Script {script_key} not found in SC.script_runners")
        return runner

    def get_control_panel(self) -> 'ControlPanel':
        """Return the shared :class:`ControlPanel`, creating it once per session."""
        if not hasattr(self, '_control_panel'):
            self._control_panel = ControlPanel()
        return self._control_panel
    
    def get_task_item(self , task_id : str | None = None) -> TaskItem | None:
        """Look up *task_id* in the queue; returns ``None`` if not found."""
        if self.task_queue is None or task_id is None:
            return None
        return self.task_queue.get(task_id)
    
    def get_latest_task_item(self , script_key : str | None = None) -> TaskItem | None:
        """Return the most-recently created task, optionally filtered by *script_key*."""
        return self.task_queue.latest(script_key)
        
    def clear_report_placeholder(self) -> None:
        """Blank out the task-report placeholder widget if it exists."""
        if 'task_report_placeholder' in st.session_state:
            with st.session_state['task_report_placeholder']:
                st.write('')

    def call_report_placeholder(self) -> Any:
        """Create (or reuse) the report placeholder and return it."""
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
    
    def get_latest_queue(self , num : int = 10) -> dict[str, TaskItem]:
        """Return the most-recent *num* tasks from the queue."""
        return self.task_queue.latest_n(num)
    
    def add_global_settings(self , params : dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge global sidebar toggles (verbosity, email, silent mode) into *params*.

        Args:
            params: Existing parameter dict; ``None`` is treated as ``{}``.

        Returns:
            New dict with global settings applied.
        """
        params = {**params} if params else {}
        params['max_vb'] = st.session_state.get('global-settings-max-vb' , False)
        if st.session_state.get('global-settings-disable-email' , False):
            params['email'] = False
        if st.session_state.get('global-settings-silent-run' , False):
            params['mode'] = 'os'
        return params

    def get_script_runner_cmd(self , runner : ScriptRunner | None , params : dict[str, Any] | None , 
                              operation_txt = True):
        """preview runner cmd"""
        if runner is None: 
            return None
        params = self.add_global_settings(params)
        cmd = runner.preview_cmd(**params)
        if operation_txt:
            run_text = 'run' if 'mode' not in params else f'{params["mode"]} run'
            cmd = f":blue[**{run_text.title()}**]: {cmd}"
        return cmd
    
    def get_script_runner_validity(self , params : dict[str, Any] | None) -> bool:
        """Return ``True`` if all required parameters in the current form have values.

        Args:
            params: Current param dict from :attr:`param_inputs_form`.
        """
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
            self.queue_last_action = f"Log Clear Success" , True
        def on_abort():
            self.queue_last_action = f"Log Clear Aborted" , False
        action_confirmation(on_confirm , on_abort , title = "Are You Sure about Clearing Logs (This Action is Irreversible)?")

    @universal_action
    def click_queue_sync(self):
        """click task queue sync"""
        self.task_queue.sync()
        self.queue_last_action = f"Queue Manually Synced at {datetime.now().strftime('%H:%M:%S')}" , True
   
    @universal_action
    def click_queue_refresh(self):
        """click task queue refresh"""
        self.task_queue.refresh()
        self.queue_last_action = f"Queue Manually Refreshed at {datetime.now().strftime('%H:%M:%S')}" , True

    @universal_action
    def click_queue_clean(self):
        """click task queue refresh confirmation"""
        items = [item for item in self.task_queue.values() if item.is_error or item.is_killed]
        [self.task_queue.remove(item) for item in items]
        self.queue_last_action = f"Queue Cleaned All Error Tasks at {datetime.now().strftime('%H:%M:%S')}" , True

    @universal_action
    def click_queue_delist_all(self):
        """click task queue delist all"""
        self.task_queue.clear_queue_only()
        self.queue_last_action = f"Entire Queue Delisted" , True

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

    @universal_action
    def click_queue_delist_item(self , item : TaskItem):
        """click task queue delist item"""
        self.task_queue.delist(item)
        self.queue_last_action = f"Delist Success: {item.id}" , True

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
        
    @universal_action
    def click_script_runner_run(self , runner : ScriptRunner , params : dict[str, Any] | None):
        """click run button"""
        params = self.add_global_settings(params)
        item = runner.build_task(self.task_queue , **params)

        self.queue_last_action = f"Add to Queue: {item.id}" , True
        self.current_task_item = item.id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

        item.run_script(from_workspace='Interactive Tasks')

    def click_file_preview(self , path : Path):
        """click file previewer"""
        Logger.stdout('click file previewer')
        if self.running_report_file_previewer == path:
            self.running_report_file_previewer = None
        else:   
            self.running_report_file_previewer = path   

    def click_file_download(self , path : Path):
        """click file previewer"""
        # TODO: things to do before download

    @universal_action
    def click_show_complete_report(self , item : TaskItem):
        """click show complete report"""
        self.current_task_item = item.id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

    @universal_action
    def click_item_choose_select(self , item : TaskItem):
        """click choose task item"""
        new_id = item.id if self.current_task_item != item.id else None
        
        self.current_task_item = new_id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

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


class ControlPanelButton(ABC):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ''
    icon : str = ''
    title : str = ''

    @abstractmethod
    def button(self , script_key : str | None = None) -> None:
        """Render the Streamlit button widget for this action.

        Args:
            script_key: The currently active script key, or ``None`` when on
                an intro page.
        """
        ...

    def refresh(self , *args , **kwargs) -> None:
        """Redraw the button with updated state (override in subclasses as needed)."""
        pass

    def show(self , script_key : str | None = None) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        if self.key not in st.session_state:
            st.session_state[self.key] = st.empty()
        with st.session_state[self.key]:
            with st.container():
                self.button(script_key = script_key)
                self.print_title()

    def print_title(self) -> None:
        """Render the small capitalised label below the button icon."""
        body = f"""
        <div style="
            margin-bottom: 0px;
            margin-top: -10px;
            padding: 0 0 20px 0;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        ">{self.title.upper()}</div>
        """       
        st.markdown(body , unsafe_allow_html = True)

class ScriptRunnerRunButton(ControlPanelButton):
    """Button that submits the current script to the task queue.

    Rendered as disabled (greyed) when no script is selected or required
    parameters are missing; enabled (green) otherwise.
    """
    key = f"script-runner-run"
    icon = f":material/mode_off_on:"
    title = f"Run Script"

    def button(self , script_key : str | None = None):
        help = f"Please Choose a Script to Run First" if script_key is None else f"Please Fill Required Parameters"
        st.button(self.icon, key=f'{self.key}-disabled' , help = help)

    def refresh(self , runner : ScriptRunner):
        with st.session_state[self.key]:
            if SC.param_inputs_form is None:
                raise ValueError("ParamInputsForm is not initiated")
            params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
            
            if SC.get_script_runner_validity(params):
                disabled = False
                preview_cmd = SC.get_script_runner_cmd(runner , params)
                if preview_cmd: 
                    help_text = preview_cmd
                else:
                    help_text = f"Parameters valid, run {runner.script_key}"
                button_key = f"{self.key}-enabled-{runner.script_key}"
            else:
                disabled = True
                help_text = f"Parameters invalid, please check required ones"
                button_key = f"{self.key}-disabled-{runner.script_key}"

            with st.container():
                st.button(self.icon, key=button_key , 
                        help = help_text , disabled = disabled , 
                        on_click = SC.click_script_runner_run , args = (runner, params)) 
                self.print_title()

class GlobalScriptLatestTaskButton(ControlPanelButton):
    """Button that navigates to the latest task across all scripts."""
    key = f"global-script-latest-task"
    icon = f":material/reply_all:"
    title = f"Latest for All"

    def button(self , script_key : str | None = None):
        item = SC.get_latest_task_item()
        if item is None:
            st.button(self.icon, key=f"{self.key}-disabled" , 
                    help = "Please Run a Task First" , disabled = True)
        else:
            if st.button(self.icon, key=f"{self.key}-enabled-{item.id}" , 
                        help = f":blue[**Show Latest Task**]: {item.id}" , 
                        on_click = SC.click_show_complete_report , args = (item,) ,
                        disabled = False):
                if SC.current_page_name != repr(item.script_key):
                    st.switch_page(item.page_url)
                else:
                    #from .script_detail import show_report_main
                    #show_report_main(SC.get_script_runner(item.script_key))
                    st.rerun()

class CurrentScriptLatestTaskButton(ControlPanelButton):
    """Button that shows the latest task for the currently displayed script."""
    key = f"current-script-latest-task"
    icon = f":material/reply:"
    title = f"Current Latest"

    def button(self , script_key : str | None = None):
        item = SC.get_latest_task_item(script_key) if script_key is not None else None
        if item is None:
            st.button(self.icon, key=f"{self.key}-disabled" , 
                        help = "Please Run a Task of This Script First" if script_key is not None else "Please Choose a Script First" , disabled = True)
        else:
            if st.button(self.icon, key=f"{self.key}-enabled-{item.id}" , 
                        help = f":blue[**Show Latest Task of This Script**]: {item.id}" , 
                        on_click = SC.click_show_complete_report , args = (item,) ,
                        disabled = False):
                #from .script_detail import show_report_main
                #show_report_main(SC.get_script_runner(item.script_key))
                st.rerun()

class ControlRefreshInteractiveButton(ControlPanelButton):
    """Button that regenerates all script-detail pages and reinitialises the session."""
    key = f"control-refresh-interactive"
    icon = f":material/refresh:"
    title = f"Refresh All"

    def button(self , script_key : str | None = None):
        st.button(self.icon, key=f"{self.key}-enabled" , help = "Refresh Task Queue / Options / Scripts" , 
                  on_click = self.refresh_all , disabled = False)
 
    def refresh_all(self):
        from src.interactive.main.util.page import remake_all_script_detail_files
        with st.spinner("Refreshing..."):
            with Proj.silence:
                Options.update()
                remake_all_script_detail_files()
        SC.rerun()
        st.rerun()

class ControlGitClearPullButton(ControlPanelButton):
    """Button that resets local changes and pulls the latest code from remote.

    Disabled automatically on coding platforms (``MACHINE.platform_coding``).
    """
    key = f"control-git-clear-pull"
    icon = f":material/cloud_download:"
    title = f"Git Pull"

    def button(self , script_key : str | None = None):
        if MACHINE.platform_coding:
            st.button(self.icon, key=f"{self.key}-disabled" , help = f"Git Pull is not available on coding platform {MACHINE.name}" , disabled = True)
        else:
            st.button(self.icon, key=f"{self.key}-enabled" , help = "Clear Local Changes and Pull Latest Code" , disabled = False, on_click = self.clear_git_pull)
        
    def clear_git_pull(self):
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        else:
            import shutil
            from src.proj import PATH , Logger

            subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
            subprocess.run(['git', 'clean', '-fd'], check=True)
            subprocess.run(['git', 'pull'], check=True)
            
            for folder in [*PATH.main.joinpath('src').rglob('*/') , *PATH.main.joinpath('configs').rglob('*/')][::-1]:
                if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
                    subfiles = [x for x in folder.rglob('*') if x.is_file()]
                    if not len(subfiles):
                        Logger.stdout(f"Removing empty folder: {folder}")
                        folder.rmdir()
                    else:
                        if all([x.suffix == '.pyc' for x in subfiles]):
                            Logger.stdout(f"Removing folder with only pyc files: {folder}")
                            shutil.rmtree(folder)
                        else:
                            Logger.error(f"Error removing folder: {folder}:")
                            Logger.error(f"Subfiles: {subfiles}")
            Logger.success("Git Pull Finished")

class ControlPanel:
    """Horizontal action bar rendered at the top of every app page.

    Contains a fixed set of :class:`ControlPanelButton` instances plus a
    settings popover for global run toggles (verbosity, email, silent mode).
    """
    control_panel_key = "page-control-panel"
    buttons : dict[str, ControlPanelButton] = {
        'script-runner-run' : ScriptRunnerRunButton(),
        'global-script-latest-task' : GlobalScriptLatestTaskButton(),
        'current-script-latest-task' : CurrentScriptLatestTaskButton(),
        'control-refresh-interactive' : ControlRefreshInteractiveButton(),
        'control-git-clear-pull' : ControlGitClearPullButton(),
    }
    
    def show(self , script_key : str | None = None) -> None:
        """Render the full control panel (buttons + settings popover).

        Args:
            script_key: Passed through to each button so they can
                enable/disable themselves based on whether a script is active.
        """
        with st.container(key = self.control_panel_key):
            columns = st.columns([1,10,1] , gap = 'small' , vertical_alignment = 'center')
            _ , buttons , settings = columns
            with buttons.container(key = f"{self.control_panel_key}-buttons"):
                self.show_buttons(script_key = script_key)
            with settings.container(key = f"{self.control_panel_key}-settings"):
                self.show_settings()

    def show_buttons(self , script_key : str | None = None) -> None:
        """Lay out one column per button and call each button's :meth:`show`."""
        cols = st.columns(len(self.buttons) , gap = 'small' , vertical_alignment = 'center')
        for col , button in zip(cols, self.buttons.values()):
            with col:
                button.show(script_key = script_key)

    def show_settings(self) -> None:
        """Render the settings gear popover with global run toggles."""
        with st.popover('**:material/settings:**'):
            st.toggle('**:blue[Max Verbosity]**', value=False , key = 'global-settings-max-vb' , 
                    help="""Should use max verbosity or min? Not selected will use default.""")
            st.toggle('**:blue[Disable Email]**', value=False , key = 'global-settings-disable-email'  , 
                    help="""If email after the script is complete? Not selected will use script header value.""")
            st.toggle("**:blue[Silent Run]**", value=False , key = 'global-settings-silent-run'  , 
                    help="""Should the script run silently? Not selected will use script header value.""")
        
          
   
SC = SessionControl()