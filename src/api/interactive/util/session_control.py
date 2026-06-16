"""Session state, action callbacks.

Key objects:

* :class:`SessionControl` (``SC``) — dataclass-based singleton that owns the
  task queue, script runners, param cache, and all button click handlers.
  Instantiated once at module import as the module-level ``SC``.
* :func:`queue_refresh_trigger` — decorator that triggers task queue refresh after
  every callback.
"""
from __future__ import annotations
import time
import streamlit as st

from dataclasses import dataclass , field
from datetime import datetime
from functools import wraps , cached_property
from pathlib import Path
from typing import Any , ClassVar  , Literal
from collections.abc import Callable

from src.proj import PATH , Const , Logger
from src.api.util.backend import TaskQueue , TaskItem , TaskDatabase , ScriptRunner , PathItem
from src.api.util.st_frontend import YAMLFileEditorState , action_confirmation , ParamCache

__all__ = ['SC' , 'SessionControl']

@st.cache_resource
def get_cached_task_db() -> TaskDatabase:
    """get cached task database manager"""
    return TaskDatabase()

_poll_fragment_cache: dict[tuple[int, float], Callable[[], None]] = {}


def _task_queue_poll_fragment(interval: float, epoch: int) -> Callable[[], None]:
    """Return a ``run_every`` fragment for backend queue polling (epoch-scoped).

    A new fragment function is created per *epoch* so git pull can retire the
    previous auto-rerun schedule without leaving a permanently active timer.
    """
    cache_key = (epoch, interval)
    if cache_key not in _poll_fragment_cache:
        _poll_fragment_cache.clear()

        @st.fragment(run_every=interval)
        def _poll() -> None:
            if st.session_state.get('backend_refresh_epoch', 0) != epoch:
                return
            sc = SessionControl._instance
            if sc is None or sc.task_queue is None:
                return
            if sc.task_queue.refresh(backend_only=True):
                st.rerun()

        _poll_fragment_cache[cache_key] = _poll
    return _poll_fragment_cache[cache_key]


def queue_refresh_trigger(func : Callable) -> Callable:
    """Decorator: trigger task queue refresh after every UI callback.

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
    current_script : str | None = None
    current_task_item : str | None = None

    # for task queue page
    queue_last_action : tuple[str, bool] | None = None
    running_report_queue : str | None = None

    # for task detail page
    running_report_init : bool = True
    running_report_file_previewer : Path | None = None

    script_params_cache : ParamCache = field(default_factory=ParamCache)

    # for config editor page
    config_editor_state : YAMLFileEditorState | None = None

    # for api console page
    api_endpoint_selected : str | None = None

    placeholders : dict[str, Any] = field(default_factory=dict)

    _instance : ClassVar[SessionControl | None] = None

    @property
    def current_runner(self) -> ScriptRunner | None:
        """get current runner"""
        if self.current_script is None:
            return None
        return self.get_script_runner(self.current_script)
    
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

    def bump_backend_refresh_epoch(self) -> None:
        """Invalidate periodic backend-refresh timers after disruptive reloads (e.g. git pull)."""
        st.session_state['backend_refresh_epoch'] = st.session_state.get('backend_refresh_epoch', 0) + 1

    def _should_poll_task_queue_backend(self) -> bool:
        if self.task_queue is None:
            return False
        if self.task_queue.count('running') > 0:
            return True
        return bool(self.task_db.get_backend_updated_tasks())

    def wrap_page(self , page_name : str) -> Callable:
        """wrap page"""
        def wrapper(func : Callable):
            @wraps(func)
            def inner_func(*args , **kwargs):
                page = self.page_header(page_name)
                if page is None:
                    return
                func(*args , **kwargs)
                self.task_queue_backend_refresh()
                interval = Const.Pref.interactive.get('task_queue_backend_refresh_interval', 5)
                if interval > 0 and self._should_poll_task_queue_backend():
                    epoch = st.session_state.get('backend_refresh_epoch', 0)
                    _task_queue_poll_fragment(interval, epoch)()
            return inner_func
        return wrapper

    @cached_property
    def intro_pages(self) -> dict:
        """get intro pages"""
        from src.api.interactive.util.page import intro_pages
        return intro_pages()

    @cached_property
    def script_pages(self) -> dict:
        """get script pages"""
        from src.api.interactive.util.page import script_pages
        return script_pages()

    def get_page(self , page_name : str) -> dict:
        """get page"""
        if page_name in self.intro_pages:
            return self.intro_pages[page_name]
        else:
            from src.api.interactive.util.page import get_page
            return get_page(page_name)

    def consume_pending_pull_and_run(self) -> None:
        """Run a script queued by :class:`ControlGitClearPullRunButton` after git pull.

        Git pull may reload the Streamlit app mid-callback; the pending payload
        in ``st.session_state`` survives that reload.
        """
        pending = st.session_state.pop('pending_pull_and_run', None)
        if pending is None:
            return
        runner = self.get_script_runner(pending['script_key'])
        self.click_script_runner_run(runner, pending.get('params'))

    @queue_refresh_trigger
    def page_header(self , page_name : str , type : Literal['intro' , 'script'] = 'intro'):
        """Register the active page and render the shared header and control panel.

        Called at the top of every page's ``main()`` function.  Sets the current
        page in session state, notifies :class:`SessionControl`, then renders the
        coloured icon + rainbow title header followed by the control panel row.

        Args:
            page_name: The intro page name (e.g. ``'home'``) or script key
                (e.g. ``'4_train/1_train_model.py'``).
            type: ``'intro'`` for intro pages, ``'script'`` for script pages.
        Returns:
            The page metadata dict, or ``None`` if the page is not found.
        """
        """Store ``key`` as the active page name in Streamlit session state."""
        self.consume_pending_pull_and_run()
        st.session_state["current_page"] = page_name
        self.current_page_name = page_name
        self.current_script = None if page_name in self.intro_pages else page_name
        self_page = self.get_page(page_name) 
        
        st.header(f"*_:red[{self_page['icon']}] :rainbow[{self_page['head']}]_*" , help = self_page['help'])
        self.control_panel.show(self.current_script)
        return self_page
    
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

    @cached_property
    def control_panel(self):
        """get control panel"""
        from src.api.interactive.util.control_panel import ControlPanel
        return ControlPanel()

    def refresh_control_panel(self , runner : ScriptRunner) -> None:
        """refresh control panel buttons"""
        self.control_panel.refresh_buttons(runner)
    
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
    
    def get_filtered_queue(self) -> dict[str, TaskItem]:
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
        params['debug_mode'] = st.session_state.get('global-settings-debug-mode' , False)
        params['max_vb'] = st.session_state.get('global-settings-max-vb' , False)
        if st.session_state.get('global-settings-disable-email' , False):
            params['email'] = False
        if st.session_state.get('global-settings-silent-run' , False):
            params['mode'] = 'os'
        return params

    def get_script_runner_cmd(
        self , runner : ScriptRunner | None , params : dict[str, Any] | None , 
        operation_txt = True
    ) -> str | None:
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

    def set_param_inputs_form(self , param_inputs_form):
        """set param inputs form"""
        from src.api.util.st_frontend import ParamInputsForm
        if not isinstance(param_inputs_form, ParamInputsForm):
            Logger.error(f"param inputs form is not a ParamInputsForm: {param_inputs_form.__class__.__name__}" , indent = 1 , vb_level = 2)
            raise ValueError("param inputs form is not a ParamInputsForm")
        self.param_inputs_form = param_inputs_form
   
    @queue_refresh_trigger
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

    @queue_refresh_trigger
    def click_log_clear_confirmation(self):
        """click log clear confirmation"""
        def on_confirm():
            self.queue_last_action = f"Log Clear Success" , True
        def on_abort():
            self.queue_last_action = f"Log Clear Aborted" , False
        action_confirmation(on_confirm , on_abort , title = "Are You Sure about Clearing Logs (This Action is Irreversible)?")

    @queue_refresh_trigger
    def click_queue_sync(self):
        """click task queue sync"""
        self.task_queue.sync()
        self.queue_last_action = f"Queue Manually Synced at {datetime.now().strftime('%H:%M:%S')}" , True
   
    @queue_refresh_trigger
    def click_queue_refresh(self):
        """click task queue refresh"""
        self.task_queue.refresh()
        self.queue_last_action = f"Queue Manually Refreshed at {datetime.now().strftime('%H:%M:%S')}" , True

    @queue_refresh_trigger
    def click_queue_clean(self):
        """click task queue refresh confirmation"""
        items = [item for item in self.task_queue.values() if item.is_error or item.is_killed]
        [self.task_queue.remove(item) for item in items]
        self.queue_last_action = f"Queue Cleaned All Error Tasks at {datetime.now().strftime('%H:%M:%S')}" , True

    @queue_refresh_trigger
    def click_queue_delist_all(self):
        """click task queue delist all"""
        self.task_queue.clear_queue_only()
        self.queue_last_action = f"Entire Queue Delisted" , True

    @queue_refresh_trigger
    def click_queue_remove_all(self):
        """click task queue refresh confirmation"""
        def on_confirm():
            self.task_queue.clear()
            self.queue_last_action = f"Entire Queue Removed Success" , True
        def on_abort():
            self.queue_last_action = f"Entire Queue Removal Aborted" , False
        action_confirmation(on_confirm , on_abort , 
                            title = "Are You Sure about Removing All Tasks in Queue (Will be Auto Backuped)?")

    @queue_refresh_trigger
    def click_queue_delist_item(self , item : TaskItem):
        """click task queue delist item"""
        self.task_queue.delist(item)
        self.queue_last_action = f"Delist Success: {item.id}" , True

    @queue_refresh_trigger
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

    @queue_refresh_trigger
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
        
    @queue_refresh_trigger
    def click_script_runner_filter(self , runner : ScriptRunner):
        """click script runner filter"""
        st.session_state['task-filter-status'] = 'All'
        st.session_state['task-filter-source'] = 'All'
        st.session_state['task-filter-path-folder'] = []
        st.session_state['task-filter-path-file'] = [runner.path.path]
        
    @queue_refresh_trigger
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
        Logger.stdout('click file previewer' , indent = 1 , vb_level = 2)
        if self.running_report_file_previewer == path:
            self.running_report_file_previewer = None
        else:   
            self.running_report_file_previewer = path   

    def click_file_download(self , path : Path):
        """click file previewer"""
        # TODO: things to do before download

    @queue_refresh_trigger
    def click_show_complete_report(self , item : TaskItem):
        """click show complete report"""
        self.current_task_item = item.id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

    @queue_refresh_trigger
    def click_item_choose_select(self , item : TaskItem):
        """click choose task item"""
        new_id = item.id if self.current_task_item != item.id else None
        
        self.current_task_item = new_id
        self.clear_report_placeholder()
        self.running_report_init = True
        self.running_report_file_previewer = None

    @queue_refresh_trigger
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
    def wait_until_completion(item : TaskItem , starting_timeout : int = Const.Pref.interactive.get('task_starting_timeout' , 20)):
        """wait for complete"""
        return item.wait_until_completion(starting_timeout)

    def task_queue_backend_refresh(self) -> None:
        """Sync task status from the database; rerun the page when anything changed."""
        changed = self.task_queue.refresh(backend_only=True)
        if changed:
            st.success(f"Task Queue Backend Refreshed: {changed}")
            time.sleep(1)
            st.rerun()

SC = SessionControl()