__version__ = '0.1.0'

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

import os, platform, subprocess, yaml, re, time, base64, glob, json, signal , torch, logging
import streamlit as st
import streamlit.components.v1 as components
import psutil
import pandas as pd
from dataclasses import dataclass , field , asdict
from functools import partial
from streamlit_autorefresh import st_autorefresh
from streamlit.delta_generator import DeltaGenerator

from typing import Any, Literal, Callable
from pathlib import Path
from datetime import datetime

from src_runs.util import terminal_cmd , check_process_status , kill_process
from src_runs.util.st_backend import PathItem , TaskItem , TaskQueue , ScriptRunner , ExitMessenger , ScriptParamInput

BASE_DIR = Path('src_runs').absolute()
AUTO_REFRESH_INTERVAL = 2000 # 10 seconds

from src_runs.util.st_backend import logger

if AUTO_REFRESH_INTERVAL:
    st_autorefresh(interval=AUTO_REFRESH_INTERVAL, key="autorefresh-example")

@dataclass
class SessionControl:
    """session control"""
    current_script_runner : str | None = None
    current_task_item : str | None = None
    
    running_report_queue : str | None = None
    running_report_main : str | None = None
    running_report_main_cleared : bool = False

    placeholder_queue_item : dict[str, Any] = field(default_factory=dict)
    script_params_cache : dict[str, dict[str, dict[str , Any]]] = field(default_factory=dict)
    
    def to_go(self , obj : ScriptRunner):
        return all(self.script_params_cache.get(obj.script_key, {}).get('valid', {}).values())
    
def session_control() -> SessionControl:
    if 'session_control' not in st.session_state:
        st.session_state.session_control = SessionControl()
    return st.session_state.session_control

def page_config():
    st.set_page_config(
        page_title="Script Runner",
        page_icon=":material/rocket_launch:",
        layout=None,
        initial_sidebar_state="collapsed"
    )
    st.title(f":rainbow[:material/rocket_launch: Script Runner (_v{__version__}_)]")
      
def page_css():
    st.markdown("""
    <style>
    h1 {
        font-size: 48px !important;
        font-weight: 900 !important;
        padding: 10px !important;
        border-bottom: 2px solid #1E90FF !important;
    }
    .stButton > button {
        padding: 2px 6px;
        font-size: 12px;
        line-height: 1.0;
        min-height: 18px !important;
        display: flex;
        justify-content: center;
        margin: 0px !important;
    }
    .stCaptionContainer {
        margin: -10px !important;
    }
    .stSelectbox {
        width: 100% !important;
    }
    .stSelectbox > div > div {
        height: 28px;
        width: 100%;
    }
    .stSelectbox > div > div > div {
        align-self: center;
    }
    .stTextInput {
        width: 100%;
    }
    .stTextInput > div {
        height: 28px;
    }
    .stTextInput > div > div {
        align-self: center !important;
    }
    .stNumberInput {
        width: 100%;
    }
    .stNumberInput > div {
        height: 28px;
        width: 100%;
    }
    .stNumberInput > div > div {
        height: 28px;
        align-self: center !important;
    }
    .stNumberInput button {
        width: 20px !important;
        height: 170% !important;
        min-height: 28px !important;
        align-self: flex-end !important;
        margin-top: 0px !important;
    }
    .element-container {
        margin-bottom: 0px;
        display: flex;
        align-items: center;
    }
    .stMarkdown {
        line-height: 1.0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stMarkdown p {
        margin-top: 0px;
        margin-bottom: 0px;
        line-height: 1.0 !important;
    }
    .stMetric div {
        font-size: 14px !important;
    }
    .stMetric > label > div > div {
    }
    .stMetric > div {
        color: blue;
    }
    .stContainer {
        padding-top: 0px;
        padding-bottom: 0px;
    } 
    .stExpander .stElementContainer {
        margin-bottom: -10px !important;
        padding-bottom: 0px !important;
    }
    [data-testid="stExpander"] summary {
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }
    [data-testid="stCode"] code {
        font-size: 12px !important;
    }
    div[data-baseweb="notification"] {
        min-height: 18px !important;
        display: flex !important;
        line-height: 1.0 s!important;
        align-items: center;
        justify-content: right;
        font-size: 14px !important;
        padding: 0.25rem 0.5rem !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }
    div[class="st-b8 st-b9"] {
        margin-top: 0px !important;
    }
    div[data-testid="column"] {
        display: flex;
        align-items: center;
    }
    [class*="classic-remover"] button {
        height: 32px !important;
        min-height: 32px !important;
        width: 32px !important;
        background-color: red !important; 
        fill: white !important; 
        color: white !important; 
        margin: 0px !important;
    }
    [class*="st-key-task-queue-refresh"] button {
        height: 36px !important;
        min-height: 36px !important;
        width: 36px !important;
        margin-right: 0px !important;
    }
    [class*="st-key-task-queue-clear"] button {
        height: 32px !important;
        min-height: 32px !important;
        width: 32px !important;
        margin-right: -40px !important;
    }
    [class*="st-key-exit-message-clear"] button {
        height: 32px !important;    
        min-height: 32px !important;
        width: 32px !important;
        margin-right: -40px !important;
    }     
    [class*="st-key-queue-item-container"] {
        margin-bottom: -10px !important;
        padding-right: 20px !important;
    }
    [class*="st-key-queue-item"] button {
        font-size: 10px !important;
    }
    [class*="st-key-queue-item-content"] button {
        min-height: 32px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 6px !important;
    }
    [class*="st-key-script-container"] div[class="stMarkdown"] > div {
        margin: 0px !important;
    }
    [class*="st-key-script-container"] {
        margin: 0px !important;
    }
    [class*="st-key-script-container"] button {
        margin-top: 10px !important;
        margin-bottom: -10px !important;
    }
    [class*="st-key-script-container"] p {
        margin-top: 0px !important;
    }
    [class*="st-key-script-container-1"] {
        margin-left: 30px !important;
    }
    [class*="st-key-script-container-2"] {
        margin-left: 60px !important;
    }
    [class*="st-key-script-container-3"] {
        margin-left: 90px !important;
    }
    [class*="st-key-script-runner-expand"] button {
        min-width: 250px !important;
        font-weight: bold !important;
        align-items: center !important;
        align-content: center !important;
        align-self: center !important;
        justify-content: flex-start !important;
        margin: 0 !important;
    }       
    [class*="st-key-script-runner-expand"] button:hover {
        background-color: lightblue !important;
        border: 1px solid lightblue !important;
        color: white !important;
    }
    [class*="st-key-script-runner-expand"] p {
        font-size: 16px !important;
        font-weight: bold !important;
    }
    [class*="script-setting-container"] {
        margin: 0px !important;
        padding-bottom: 10px !important;
    }
    [class*="st-key-script-runner-run"] button {
        min-width: 50px !important;
        height: 50px !important;
        width: 50px !important;
        background-color: green !important;
        color: white !important;
        border-radius: 50%;
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px !important;
    }
    [class*="st-key-script-runner-run"] button:hover {
        color: white !important;
        background-color: darkgreen !important;
        border: 1px solid darkgreen !important;
    }
    [class*="st-key-script-runner-run"] p {
        font-size: 36px !important;
        font-weight: bold !important;
    }
    [class*="st-key-script-runner-run-disabled"] button{
        background-color: lightgray !important;
        color: white !important;
        border: 1px solid lightgray !important;
    }
    [class*="st-key-script-runner-run-disabled"] button:hover {
        background-color: lightgray !important;
        color: white !important;
        border: 1px solid lightgray !important;
    }
    .stElementContainer[class*="st-key-script-setting-classic-remover"] div {
        align-items: flex-start !important;
        justify-content: flex-end !important;        
    }
    .stElementContainer[class*="st-key-script-setting-classic-remover"] button {
        margin: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def show_sidebar():
    sc = session_control()
    with st.sidebar:
        st.header(":material/book: Manual" , divider = 'grey')
        st.markdown("""
        1. :green[:material/play_circle:] Click the script button to expand the parameter settings
        2. :blue[:material/app_registration:] Fill in the necessary parameters and click Run
        3. :rainbow[:material/bar_chart:] View the running report and generated files
        4. :material/description: Preview the generated HTML/PDF files
        """)
        
        st.header(":material/computer: System Info" , divider = 'grey')
        if torch.cuda.is_available():
            gpu_info = f":material/memory_alt: **GPU Memory:** {torch.cuda.memory_summary(0)}"
        elif torch.mps.is_available():
            gpu_info = f":material/memory_alt: **MPS Memory:** {torch.mps.current_allocated_memory()/1024**3:.1f} / {torch.mps.recommended_max_memory()/1024**3:.1f} GB"
        else:
            gpu_info = ":material/memory_alt: **GPU Memory:** No GPU"
        options = [
            f":material/keyboard_command_key: **OS:** {platform.system()}" , 
            f":material/memory: **Memory:** {psutil.virtual_memory().percent:.1f}%" , 
            gpu_info , 
            f":material/select_all: **CPU:** {psutil.cpu_percent():.1f}%" , 
            f":material/commit: **Python:** {sys.version.split('(')[0]}" , 
            f":material/commit: **Streamlit:** {st.__version__}" 
        ]
        st.pills("Informations" , options , key = "system-info-pills" , format_func=lambda x: f':blue[{x}]', label_visibility="hidden")
            
        st.header(":material/pending_actions: Pending Features" , divider = 'grey')
        st.warning("Open Script for sub Task List" , icon = ":material/schedule:")
        st.warning("Running Report include generated files that exported from scripts (record in json with key input)" , icon = ":material/schedule:")
        st.warning("Edit config files" , icon = ":material/schedule:")

        st.header(":material/bug_report: Debug Info" , divider = 'grey')
        expander_col , button_col = st.columns([8, 1] , vertical_alignment = "top")
        with expander_col:
            with st.expander("View queue file", expanded=False , icon = ":material/file_json:"):
                TaskQueue.refresh()
                try:
                    st.code(TaskQueue.full_content(), language='json' , )
                except Exception as e:
                    st.error(f"Error loading queue: {e}")
        with button_col:
            if st.button("", key="task-queue-clear", icon = ":material/delete:" , help = "Clear All Tasks"):
                TaskQueue.clear()

    show_queue_in_sidebar()

def show_queue_in_sidebar():
    # queue title and refresh button
    sc = session_control()

    TaskQueue.refresh()
    with st.sidebar:
    
        header_col, button_col = st.columns([8, 1] , vertical_alignment = "center")
        
        with header_col: 
            st.header(":material/event_list: Running Queue" , divider = 'grey')
        with button_col:
            if st.button("", key="task-queue-refresh", icon = ":material/directory_sync:" , 
                        help = "Refresh Queue" + (f" (every {AUTO_REFRESH_INTERVAL/1000} seconds)" if AUTO_REFRESH_INTERVAL else "")):
                # st.rerun()
                pass

        if TaskQueue.empty():
            st.info("Queue is empty, click the script below to run and it will be displayed here" , icon = ":material/queue_play_next:")
            return
        st.caption(f":rainbow[:material/bar_chart:] {TaskQueue.status_message()}")
        st.markdown("")
        # Show queue items
        for item in TaskQueue.values():
            placeholder = st.empty()
            container = placeholder.container(key = f"queue-item-container-{item.id}")
            with container:
                content_col , remove_col = st.columns([9, 1] , gap = "small" , vertical_alignment = "center")
                    
                with content_col:
                    help_text = '|'.join([f"Status: {item.status}" , f"Duration: {item.duration_str} Secs", f"PID: {item.pid}"])
                    st.button(f"{item.icon} {item.button_str}",  help=help_text , key=f"queue-item-content-{item.id}" , 
                            use_container_width=True , on_click = click_queue_item , args = (item,))
                
                with remove_col:
                    if remove_button := st.button(":material/close:", key=f"queue-item-classic-remover-{item.id}", help="Remove/Terminate", type="secondary"):
                        if item.kill():
                            st.success(f"✅ Process {item.pid} terminated")
                        else:
                            st.warning("⚠️ Terminate process failed")
                        TaskQueue.remove(item)

                if remove_button:
                    st.success(f"✅ Removed from queue: {item.script}")
                    time.sleep(0.5)  # small delay to ensure status update
                    st.rerun()
                
                if sc.running_report_queue is None or sc.running_report_queue != item.id:
                    continue
            
                status_text = f'Running Report {item.status_state.title()}'
                status = st.status(status_text , state = item.status_state , expanded = True)

                with status:
                    col_config = {
                        'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                        'Value': st.column_config.TextColumn(width="large", help='Value of the item')
                    }

                    st.dataframe(item.dataframe() , row_height = 20 , column_config = col_config)
                    wait_for_complete(item)
                    st.success(f'Script Completed' , icon = ":material/add_task:")
                    #if st.button("Show complete Report", key=f"show-complete-report-{item.id}"):
                    #    show_report_main(item)


def click_queue_item(item : TaskItem):
    """click queue item"""
    sc = session_control()
    if sc.running_report_queue is not None and sc.running_report_queue == item.id:
        sc.running_report_queue = None
    else:
        sc.running_report_queue = item.id

def click_script_runner_expand(runner : ScriptRunner):
    """click script toggle"""
    sc = session_control()
    if sc.current_script_runner is not None and runner == sc.current_script_runner:
        sc.current_script_runner = None
    else:
        sc.current_script_runner = runner.id
    # st.rerun()

def click_run_button(runner : ScriptRunner , params : dict[str, Any]):
    """click queue item"""
    sc = session_control()
    run_params = {
        'email': int(runner.header.email),
        'close_after_run': bool(runner.header.close_after_run)
    }
    run_params.update(params)
    item = runner.run_script(**run_params)
    sc.current_task_item = item.id
    TaskQueue.refresh()
    if sc.running_report_main != item.id:
        sc.running_report_main = item.id
        sc.running_report_main_cleared = False
    # st.rerun()

def wait_for_complete(item : TaskItem , running_timeout : int = 20):
    """wait for complete"""
    while True:
        item.refresh()
        if item.status in ['complete' , 'error']:
            break
        if item.status == 'starting':
            running_timeout -= 1
            if running_timeout <= 0:
                raise RuntimeError(f'Script {item.script} running timeout! Still starting')
        time.sleep(1)
    return item

def preview_text_file(file_path):
    """preview text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        st.code(text_content , language=None)
    except Exception as e:
        st.error(f"Cannot preview text file: {str(e)}")

def preview_html_file(file_path):
    """preview HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Cannot preview HTML file: {str(e)}")

def preview_pdf_file(file_path):
    """preview PDF file"""
    try:
        with open(file_path, 'rb') as f:
            pdf_data = f.read()
        
        # use base64 to encode PDF
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{pdf_base64}" 
                width="100%" height="600px" type="application/pdf">
        </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Cannot preview PDF file: {str(e)}")

def show_session_control():
    """show session control"""
    sc = session_control()
    st.write(sc)

def show_folder(folder_path: Path | str = BASE_DIR, min_level: int = 0, max_level: int = 2):
    """show folder content recursively"""
    items = PathItem.iter_folder(folder_path, min_level = min_level, max_level = max_level)
    
    for item in items:
        if item.is_dir:
            st.markdown(f"""<hr style="
                        height:1px;
                        border-width:0;
                        background-color:lightgrey;
                        padding:{4 / (item.level+2)}px {400-item.level*15}px;
                        margin-top:0px;
                        margin-bottom:0px;
                        margin-left:{(item.level-1)*30}px;
                    "><div class="blue-line">   </div>""", unsafe_allow_html=True)
            folder_name = re.sub(r'^\d+_', '', item.name).replace('_', ' ').title()
            html_style = f"font-size:20px;font-weight:bold;margin-top:10px;margin-bottom:10px;margin-left:{(item.level-1)*30}px;"
            st.markdown(f"<div style='{html_style}'>📂 {folder_name}</div>" , unsafe_allow_html=True)
        elif item.level > 0:
            show_script_runner(item.script_runner())

def show_script_runner(runner: ScriptRunner):
    """show single script runner"""
    sc = session_control()
    with st.container(key = f"script-container-{runner.level}-{runner.script_key}"):
        button_col, content_col = st.columns([1, 1] , gap = 'large' , vertical_alignment = "center")
        with button_col:
            _status = ':no_entry:' if runner.header.disabled else ':arrow_forward:'
            _is_current = runner == sc.current_script_runner
            button_text = ' '.join([':arrow_down_small:' if _is_current else _status, ':snake:', runner.desc])
            st.button(f"**{button_text}**" , key=f"script-runner-expand-{runner.script_key}" , help = str(runner.script) ,
                       on_click = click_script_runner_expand , args = (runner,))

        with content_col:
            # st.markdown(f":speech_balloon: {runner.content}" , help = runner.todo)
            html_style = f"font-size:13px;line-height:1.5;margin-left:{-(runner.level-1)*30}px;margin-top:0px;margin-bottom:0px;align-self: flex-start;"
            st.caption(f"<div style='{html_style}'>💬 {runner.content}</div>" , help = runner.todo , unsafe_allow_html=True)

        if sc.current_script_runner is None or runner != sc.current_script_runner: return
        show_script_details(runner)

    # st.rerun()

def show_script_details(runner: ScriptRunner):
    """show script details"""
    sc = session_control()
    
    if todo := runner.header.todo:
        st.info(f":material/pending_actions: {todo}")
    if runner.disabled:
        st.error(f":material/disabled_by_default: This script is disabled")
        return
    
    with st.container(key = f"script-setting-container-{runner.script_key}" , border = True):
        sc = session_control()
        param_inputs = runner.header.get_param_inputs()
        settings_col , collapse_col = st.columns([1, 1] , vertical_alignment = "center")
        with settings_col:
            if not param_inputs:
                st.info("**No parameter settings**" , icon = ":material/settings:")
            else:
                st.info("**Parameter Settings**" , icon = ":material/settings:")

        with collapse_col:
            st.button(":material/close:", key=f"script-setting-classic-remover-{runner.script_key}", help="Collapse", type="secondary" ,
                      on_click = click_script_runner_expand , args = (runner,))                
        
        params = ParamInputsForm(runner).init_param_inputs('form')
        help_text = f"Run Script {runner.script}" if sc.to_go(runner) else f"Please check all parameters"
        button_key = "script-runner-run-" + \
                     ("enabled" if sc.to_go(runner) else "disabled") + \
                     f"-{runner.script_key}"
        st.button(":material/mode_off_on:", key=button_key , 
                  help = help_text , disabled = not sc.to_go(runner) , on_click = click_run_button , args = (runner,params))
        show_report_main(runner)

class ParamInputsForm:
    def __init__(self , runner : ScriptRunner):
        self.runner = runner
        self.param_list = [self.WidgetParamInput(runner, p) for p in runner.header.get_param_inputs()]
        self.errors = []
        self.sc = session_control()

    def init_param_inputs(self , type : Literal['customized', 'form'] = 'customized'):
        if type == 'customized':
            self.init_customized_container()
        elif type == 'form':
            self.init_form()
        else:
            raise ValueError(f"Invalid param inputs type: {type}")

    class WidgetParamInput(ScriptParamInput):
        def __init__(self , runner : ScriptRunner , param : ScriptParamInput):
            super().__init__(**param.as_dict())
            self._runner = runner
            self._param = param
            self.widget_key = self.get_widget_key(runner, param)
            self.transform = self.value_transform(param)

        @property
        def script_key(self):
            return self._runner.script_key

        @property
        def raw_value(self) -> Any:
            return st.session_state[self.widget_key]
        
        @property
        def param_value(self):
            return self.transform(self.raw_value)
        
        def is_valid(self):
            return self._param.is_valid(self.param_value)
        
        def error_message(self):
            return self._param.error_message(self.param_value)
        
        @classmethod
        def get_widget_key(cls , runner : ScriptRunner , param : ScriptParamInput):
            return f"script-param-{runner.script_key}-{param.name}"
        
        @classmethod
        def value_transform(cls , param : ScriptParamInput):
            ptype = param.ptype
            if isinstance(ptype, list):
                options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
                return cls.raw_option([None] + ptype, options)
            elif ptype == str:
                return lambda x: (x.strip() if x is not None else None)
            elif ptype == bool:
                return lambda x: None if x is None or x == 'Choose an option' else bool(x)
            elif ptype == int:
                return lambda x: None if x is None else int(x)
            elif ptype == float:
                return lambda x: None if x is None else float(x)
            else:
                raise ValueError(f"Unsupported param type: {ptype}")

        @classmethod
        def raw_option(cls , raw_values : list[Any], alt_values : list[Any]):
            def wrapper(alt : Any):
                """get index of value in options"""
                raw = raw_values[alt_values.index(alt)] if alt is not None else None
                return raw
            return wrapper
        
        def on_change(self):
            sc = session_control()
            if self.script_key not in sc.script_params_cache:
                sc.script_params_cache[self.script_key] = {
                    'raw': {},
                    'value': {},
                    'valid': {}
                }
            sc.script_params_cache[self.script_key]['raw'][self.name] = self.raw_value
            sc.script_params_cache[self.script_key]['value'][self.name] = self.param_value
            sc.script_params_cache[self.script_key]['valid'][self.name] = self.is_valid()

    def init_customized_container(self , num_cols : int = 3):
        num_cols = min(num_cols, len(self.param_list))
        
        for i, wp in enumerate(self.param_list):
            if i % num_cols == 0:
                param_cols = st.columns(num_cols)
            with param_cols[i % num_cols]:
                self.get_widget(
                    runner=self.runner, param=wp, 
                    on_change=self.on_widget_change, args=(wp,))
                self.on_widget_change(wp)
                if err_msg := wp.error_message():
                    st.error(err_msg , icon = ":material/error:")
        return self.sc.script_params_cache[self.runner.script_key]['value']

    def init_form(self):
        with st.form(f"ParamInputsForm-{self.runner.script_key}" , clear_on_submit = False):
            for param in self.param_list:
                self.get_widget(self.runner, param)

            if st.form_submit_button(
                "Submit" ,
                help = "Submit Parameters to Run Script" ,
            ):
                self.submit()

        return self.sc.script_params_cache[self.runner.script_key]['value']

    def validate(self):
        self.errors = []
        for wp in self.param_list:
            if err_msg := wp.error_message():
                self.errors.append(err_msg)
                
        return len(self.errors) == 0

    def submit(self):
        for wp in self.param_list: wp.on_change()
            
        if self.validate():
            st.success(f"数据处理成功！参数: {self.param_list}")
            self.process()
        else:
            st.session_state.form_errors = self.errors
            for err_msg in self.errors:
                st.error(err_msg , icon = ":material/error:")
    
    def process(self):
        ...

    @classmethod
    def get_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                   on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        
        if isinstance(ptype, list):
            func = cls.list_widget
        elif ptype == str:
            func = cls.text_widget
        elif ptype == bool:
            func = cls.bool_widget
        elif ptype == int:
            func = cls.int_widget
        elif ptype == float:
            func = cls.float_widget
        else:
            raise ValueError(f"Unsupported param type: {ptype}")
        
        return func(runner, param, on_change, args, kwargs)

    @classmethod
    def value_transform(cls , param : ScriptParamInput):
        ptype = param.ptype
        if isinstance(ptype, list):
            options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
            return cls.raw_option([None] + ptype, options)
        elif ptype == str:
            return lambda x: (x.strip() if x is not None else None)
        elif ptype == bool:
            return lambda x: None if x is None or x == 'Choose an option' else bool(x)
        elif ptype == int:
            return lambda x: None if x is None else int(x)
        elif ptype == float:
            return lambda x: None if x is None else float(x)
        else:
            raise ValueError(f"Unsupported param type: {ptype}")

    @classmethod
    def raw_option(cls , raw_values : list[Any], alt_values : list[Any]):
        def wrapper(alt : Any):
            """get index of value in options"""
            raw = raw_values[alt_values.index(alt)] if alt is not None else None
            return raw
        return wrapper
    
    @classmethod
    def get_title(cls , param : ScriptParamInput):
        return f':red[:material/asterisk: **{param.title}**]' if param.required else f'**{param.title}**'
    
    @classmethod
    def get_widget_key(cls , runner : ScriptRunner , param : ScriptParamInput):
        return f"script-param-{runner.script_key}-{param.name}"

    @classmethod
    def get_default_value(cls , runner : ScriptRunner , param : ScriptParamInput):
        widget_key = cls.get_widget_key(runner, param)
        sc = session_control()
        default_value = sc.script_params_cache.get(runner.script_key, {}).get('raw', {}).get(param.name, param.default)
        if default_value is None:
            default_value = st.session_state[widget_key] if widget_key in st.session_state else param.default
        return default_value
    
    @classmethod
    def get_widget_value(cls , runner : ScriptRunner , param : ScriptParamInput):
        widget_key = cls.get_widget_key(runner, param)
        return st.session_state[widget_key]

    @classmethod
    def list_widget(cls , runner : ScriptRunner , param : ScriptParamInput , 
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert isinstance(ptype, list) , f"Param {param.name} is not a list"
        
        widget_key = cls.get_widget_key(runner, param)
        default_value = cls.get_default_value(runner, param)
        title = cls.get_title(param)
        options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
        return st.selectbox(
            title,
            options,
            index=0 if default_value is None else options.index(default_value),
            key=widget_key ,
            on_change = on_change ,
            args = args ,
            kwargs = kwargs
        )
    
    @classmethod
    def text_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == str , f"Param {param.name} is not a string"
        widget_key = cls.get_widget_key(runner, param)
        default_value = cls.get_default_value(runner, param)
        title = cls.get_title(param)
        return st.text_input(
            title,
            value=None if default_value is None else str(default_value),
            placeholder=param.placeholder ,
            key=widget_key ,
            on_change = on_change ,
            args = args ,
            kwargs = kwargs
        )
    
    @classmethod
    def bool_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == bool , f"Param {param.name} is not a boolean"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param)
        return st.selectbox(
            title,
            ['Choose an option', True, False],
            index=0 if default_value is None else 2-bool(default_value),    
            key=widget_key ,
            on_change = on_change ,
            args = args ,
            kwargs = kwargs
        )
    
    @classmethod
    def int_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                   on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == int , f"Param {param.name} is not an integer"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param)
        return st.number_input(
            title,
            value=None if default_value is None else int(default_value),
            min_value=param.min,
            max_value=param.max,
            placeholder=param.placeholder,
            key=widget_key ,
            on_change = on_change ,
            args = args ,
            kwargs = kwargs
        )
    
    @classmethod
    def float_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == float , f"Param {param.name} is not a float"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param)
        return st.number_input(
            title,
            value=None if default_value is None else float(default_value),
            min_value=param.min,
            max_value=param.max,
            step=0.1,
            placeholder=param.placeholder,
            key=widget_key ,
            on_change = on_change ,
            args = args ,
            kwargs = kwargs
        )

    @classmethod
    def get_form_errors(cls):
        return st.session_state.form_errors
    
    @classmethod
    def on_widget_change(cls , wp : WidgetParamInput):
        wp.on_change()
        
def show_report_main(runner : ScriptRunner):
    """show complete report"""
    sc = session_control()
    item = TaskQueue.get(sc.current_task_item)
    if item is None: return
    if not item.belong_to(runner): return

    status_text = f'Running Report {item.status_state.title()}'
    status_placeholder = st.empty()
    status = status_placeholder.status(status_text , state = item.status_state , expanded = True)
       
    with status_placeholder:
        if not sc.running_report_main_cleared:
            st.write('')
            sc.running_report_main_cleared = True
            st.rerun()

        with status:
            with st.expander(":rainbow[:material/build:] **Command Details**", expanded=False):
                st.code(item.cmd , wrap_lines=True)

            script_str = f"Script [{item.format_path}] (PID: {item.pid})"
            if item and item.status == 'error':
                st.error(f'{script_str} has error' , icon = ":material/error:")
                return
            else:
                st.success(f'{script_str} started' , icon = ":material/add_task:")

            df_placeholder = st.empty()
            col_config = {
                'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                'Value': st.column_config.TextColumn(width="large", help='Value of the item')
            }
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(include_exit_info = False) , row_height = 20 , column_config = col_config)

            wait_for_complete(item)
            TaskQueue.refresh()
            
            st.success(f'{script_str} Completed' , icon = ":material/trophy:")

            with st.expander(f":rainbow[:material/fact_check:] **Exit Information**", expanded=True):
                exit_info_list = item.exit_info_list()
                for name , value in exit_info_list:
                    st.metric(f":blue-badge[**{name}**]", value)

            if item.exit_files:
                with st.expander(f":rainbow[:material/file_present:] **File Previewer**", expanded=True):
                    for file_path in item.exit_files:
                        path = Path(file_path).absolute()
                        if not path.exists(): 
                            with open(path , 'w') as f:
                                f.write('aaaaa')
                        if path.exists():
                            col1, col2, col3 = st.columns([3, 1, 1] , vertical_alignment = "center")
                            with col1:
                                st.write(f"📄 {file_path}")
                            with col2:
                                suffix = path.suffix
                                if suffix in ['.txt', '.csv', '.json' , '.log' , '.py']:
                                    if st.button("Preview", key=f"preview_{file_path}_{item.id}"):
                                        preview_text_file(path)
                                elif suffix == '.html':
                                    if st.button("Preview", key=f"preview_{file_path}_{item.id}"):
                                        preview_html_file(path)
                                elif suffix == '.pdf':
                                    if st.button("Preview", key=f"preview_{file_path}_{item.id}"):
                                        preview_pdf_file(file_path)
                            with col3:
                                try:
                                    with open(file_path, 'rb') as f:
                                        st.download_button(
                                            'Download', 
                                            data=f.read(),
                                            file_name=str(path),
                                            key=f"download_{file_path}_{item.id}"
                                        )
                                except:
                                    st.error("File read error")
                        else:
                            st.warning(f"⚠️ File not found: {file_path}")

        st.rerun()


def main():
    page_config()
    page_css()
    show_sidebar()
    show_session_control()
    show_folder()   

if __name__ == '__main__':
    main() 