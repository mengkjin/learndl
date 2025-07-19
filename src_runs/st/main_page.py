__version__ = '0.1.6'
__recommeded_explorer__ = 'chrome'

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

import platform, re, time, torch
import streamlit as st
import streamlit.components.v1 as components
import psutil
from dataclasses import dataclass , field
from streamlit_autorefresh import st_autorefresh

from typing import Any, Literal, Callable
from pathlib import Path
from datetime import datetime

from src_runs.st.backend import (
    BASE_DIR , 
    PathItem , TaskItem , TaskQueue , ScriptRunner , ScriptParamInput
)

from src_runs.st.frontend import (
    CustomCSS , FilePreviewer , ActionLogger , YAMLFileEditor , YAMLFileEditorState , ColoredText
)

if AUTO_REFRESH_INTERVAL := 0:
    st_autorefresh(interval=AUTO_REFRESH_INTERVAL, key="autorefresh-example")

PENDING_FEATURES = [
    'uvx'
]

def page_config():
    st.set_page_config(
        page_title="Script Runner",
        page_icon=":material/rocket_launch:",
        layout='wide',
        initial_sidebar_state="expanded"
    )
    st.title(f":rainbow[:material/rocket_launch: Script Runner (_v{__version__}_)]")
      
def system_info():
    if torch.cuda.is_available():
        gpu_info = f"**GPU Memory:** {torch.cuda.memory_summary(0)}"
    elif torch.backends.mps.is_available():
        if torch.__version__ >= '2.3.0':
            gpu_info = f"**MPS Memory:** {torch.mps.current_allocated_memory()/1024**3:.1f} / {torch.mps.recommended_max_memory()/1024**3:.1f} GB"
        else:
            gpu_info = f"**MPS Memory:** {torch.mps.current_allocated_memory()/1024**3:.1f} GB"
    else:
        gpu_info = "**GPU Memory:** No GPU"
    options = [
        f":material/keyboard_command_key: **OS:** {platform.system()}" , 
        f":material/memory: **Memory:** {psutil.virtual_memory().percent:.1f}%" , 
        f":material/memory_alt: {gpu_info}" , 
        f":material/select_all: **CPU:** {psutil.cpu_percent():.1f}%" , 
        f":material/commit: **Python:** {sys.version.split('(')[0]}" , 
        f":material/commit: **Streamlit:** {st.__version__}" 
    ]
    return options

@dataclass
class SessionControl:
    """session control"""
    script_runners : dict[str, ScriptRunner] = field(default_factory=dict)
    current_script_runner : str | None = None
    current_task_item : str | None = None
    script_runner_trigger_item : str | None = None
    # detail_item_expander_expanded : bool = False
    choose_task_item : str | None = None

    task_queue : TaskQueue | Any = None
    queue_last_action : tuple[str, bool] | None = None
    
    running_report_queue : str | None = None
    running_report_main : str | None = None
    running_report_main_cleared : bool = False

    running_report_file_previewer : Path | None = None

    script_params_cache : dict[str, dict[str, dict[str , Any]]] = field(default_factory=dict)
    config_editor_state : YAMLFileEditorState | None = None
    
    def __new__(cls):
        if 'session_control' not in st.session_state:
            st.session_state.session_control = super().__new__(cls)
        return st.session_state.session_control
    
    def __post_init__(self):
        self.task_queue = TaskQueue('script_runner_main' , 100)
        self.path_items = PathItem.iter_folder(BASE_DIR, min_level = 0, max_level = 2)

    def filter_task_queue(self):
        """filter task queue"""
        status_filter = st.session_state.get('queue-filter-status')
        folder_filter = st.session_state.get('queue-filter-path-folder')
        file_filter = st.session_state.get('queue-filter-path-file')
        filtered_queue = self.task_queue.filter(status = status_filter,
                                                folder = folder_filter,
                                                file = file_filter)
        return filtered_queue

    def ready_to_go(self , obj : ScriptRunner):
        return all(self.script_params_cache.get(obj.script_key, {}).get('valid', {}).values())
    
    @ActionLogger.log_action()
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

    @ActionLogger.log_action()
    def click_queue_filter_status(self):
        """click task queue filter status"""
        self.queue_last_action = f"Queue Filter Status: {st.session_state.get('queue-filter-status')}" , True

    @ActionLogger.log_action()
    def click_queue_filter_path_folder(self):
        """click task queue filter path folder"""
        folder_options = st.session_state.get('queue-filter-path-folder')
        if folder_options is not None:
            folder_options = [str(item.relative_to(BASE_DIR)) for item in folder_options]
        self.queue_last_action = f"Queue Filter Path Folder: {folder_options}" , True
        
    @ActionLogger.log_action()
    def click_queue_filter_path_file(self):
        """click task queue filter path file"""
        file_options = st.session_state.get('queue-filter-path-file')
        if file_options is not None:
            file_options = [str(item.relative_to(BASE_DIR)) for item in file_options]
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
    def click_queue_refresh(self):
        """click task queue refresh"""
        self.task_queue.refresh()
        self.queue_last_action = f"Queue Manually Refreshed at {datetime.now().strftime('%H:%M:%S')}" , True

    @ActionLogger.log_action()
    def click_queue_remove_item(self , item : TaskItem):
        """click task queue remove item"""
        if item.kill():
            self.queue_last_action = f"Remove Success: {item.id}" , True
        else:
            self.queue_last_action = f"Remove Failed: {item.id}" , False
        self.task_queue.remove(item)
        self.task_queue.save()

    @ActionLogger.log_action()
    def click_script_runner_expand(self , runner : ScriptRunner):
        """click script runner expand"""
        if self.current_script_runner is not None and self.current_script_runner == runner.script_key:
            self.current_script_runner = None
        else:
            self.current_script_runner = runner.script_key

    @ActionLogger.log_action()
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
        item = runner.run_script(queue = self.task_queue , **run_params)
        self.current_task_item = item.id
        self.queue_last_action = f"Add to Queue: {item.id}" , True
        self.task_queue.refresh()
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
        #self.detail_item_expander_expanded = True

    @ActionLogger.log_action()
    def click_item_choose_remove(self , item : TaskItem):
        """click remove task item"""
        self.task_queue.remove(item)
        self.task_queue.save()
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
    
SC = SessionControl()

def page_css():
    css = CustomCSS(add_css = ['basic' , 'special_expander' , 'classic_remover' , 'multi_select'])
    css.add("""
    .stElementContainer[class*="task-queue"] {
        div {justify-content: flex-end !important;}
        * {
            font-weight: bold !important;
            font-size: 24px !important;
        }
        button {
            width: 36px !important;
            height: 36px !important; 
        }
        &[class*="-refresh"] button {
            color: #1E88E5 !important;
            &:hover {
                background-color: #1E88E5 !important;
                color: white !important;
            }
        }  
        &[class*="-clear"] button {
            color: red !important;
            &:hover {
                background-color: red !important;
                color: white !important;
            }
        }
    }
    .stElementContainer[class*="confirm-clear"] {
        button {
            height: 36px !important;
            width: 200px !important;
            border-radius: 10px !important;
            font-size: 24px !important;
            font-weight: bold !important;
            color: white !important;
            border: none !important;
            div {font-size: 20px !important;}
        }
        &[class*="-confirm"] button {
            background-color: green !important;
            &:hover {background-color: darkgreen !important;}
        }
        &[class*="-abort"] button {
            background-color: red !important;
            &:hover {background-color: darkred !important;}
        }
    }
    .stVerticalBlock[class*="queue-item-container"] {
        margin-bottom: -10px !important;
        padding-right: 20px !important;
        
        [class*="-queue-item"] button {
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 6px !important;
            p {font-size: 16px !important;}
        }
        .stElementContainer[class*="queue-item-remover"] , .stElementContainer[class*="show-complete-report"] {
            * {
                border: none !important;
                border-radius: 10px !important;
                line-height: 32px !important;
                font-weight: bold !important;
                font-size: 24px !important;
            }     
            button {
                width: 32px !important;
                height: 32px !important;
                color: red !important;
                &:hover {
                    color: white !important;
                    background-color: red !important;
                }
            }
        }
        .stElementContainer[class*="show-complete-report"] {
            &[class*="-success"] button {
                color: green !important;
                &:hover {
                    color: white !important;
                    background-color: green !important;
                }
            }
        }     
    }
    .stVerticalBlock[class*="developer-info"] {
        button {
            min-width: 100px !important;
            height: 20px !important;
            padding: 10px 10px !important;
            border: 1px solid lightgray !important;
            border-radius: 10px !important;
            background-color: lightgray !important;
            font-weight: bold !important;
        }
    }
    .stElementContainer[class*="choose-item-select"] {
        button {
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 12px !important;
        }
        &[class*="-selected"] {
            button {
                background-color: #1E88E5 !important;
                color: white !important;
            }
            p {
                font-weight: bold !important;
            }
        }
    }
    .stVerticalBlock[class*="choose-item-remover"] {
        .stTooltipIcon {justify-content: flex-end !important;}
        .stElementContainer[class*="remover-button"] {
            * {
                border: none !important;
                border-radius: 10px !important;
                line-height: 32px !important;
                font-weight: bold !important;
                font-size: 24px !important;
            }     
            button {
                width: 32px !important;
                height: 32px !important;
                color: red !important;
                &:hover {
                    color: white !important;
                    background-color: red !important;
                }
            }
        }
    }
    .stVerticalBlock[class*="script-container"] {
        margin-bottom: -10px !important;
        div[class="stMarkdown"] > div {margin: 0px !important;}
        &[class*="-level-1"] {margin-left: 15px !important;}
        &[class*="-level-2"] {margin-left: 30px !important;}
        &[class*="-level-3"] {margin-left: 45px !important;}
        .stElementContainer[class*="-runner-expand"] {
            button {
                min-width: 250px !important;
                font-weight: bold !important;
                justify-content: flex-start !important;
                margin: 0 !important;
            }       
            p {
                font-size: 16px !important;
                font-weight: bold !important;
            }
            &[class*="-selected"] button {
                background-color: #1E88E5 !important;
                color: white !important;
            }  
        }
    } 
    .stVerticalBlock[class*="script-setting-container"] {
        margin: 0px !important;
        padding-bottom: 10px !important;
        .stElementContainer[class*="script-setting-classic-remover"] {
            div {
                align-items: flex-start !important;
                justify-content: flex-end !important; 
            }
            button {margin: 0px !important;}
        }
    }
    .stElementContainer[class*="script-runner-run"] {
        button {
            min-width: 50px !important;
            height: 50px !important;
            width: 50px !important;
            background-color: green !important;
            color: white !important;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            display: flex;
            margin: 20px !important;
            &:hover {background-color: darkgreen !important;}
        }
        p {
            font-size: 36px !important;
            font-weight: bold !important;
        }
        &[class*="-disabled"] button {
            background-color: lightgray !important;
            color: white !important;
            border: 1px solid lightgray !important;
            &:hover {
                background-color: lightgray !important;
            }
        } 
    }
    .stElementContainer[class*="file-preview"] {
        button {
            min-width: 500px !important;
            justify-content: flex-start !important;
        }
    }
    .stVerticalBlock[class*="file-download"] {
        .stElementContainer {justify-content: flex-end !important;}
    }  
    """)
    css.apply()

def show_sidebar():
    with st.sidebar:
        st.header(":material/book: Manual" , divider = 'grey')
        st.markdown("""
        1. :blue[:material/settings:] Click the script button to expand the parameter settings
        2. :green[:material/mode_off_on:] Fill in the necessary parameters and click Run
        3. :rainbow[:material/bar_chart:] View the running report and generated files
        4. :gray[:material/file_present:] Preview the generated HTML/PDF files
        """)
        
        st.header(":material/computer: System Info" , divider = 'grey')
        options = system_info()
        st.pills("Informations" , options , key = "system-info-pills" , 
                 format_func=lambda x: f':blue[{x}]', label_visibility="collapsed")
            
        st.header(":material/pending_actions: Pending Features" , divider = 'grey')
        for feature in PENDING_FEATURES:
            st.warning(feature , icon = ":material/schedule:")

    show_queue_in_sidebar()

def show_queue_in_sidebar():
    # queue title and refresh button
    with st.sidebar:
    
        header_col, button_col = st.columns([7, 2] , vertical_alignment = "center")
        
        with header_col: 
            st.header(":material/event_list: Running Queue" , divider = 'grey')
        with button_col:
            col1 , col2 = st.columns([1, 1] , gap = "small" , vertical_alignment = "center")
            with col1:  
                st.button(":material/directory_sync:", key="task-queue-refresh",  
                         help = "Refresh Queue" + (f" (every {AUTO_REFRESH_INTERVAL/1000} seconds)" if AUTO_REFRESH_INTERVAL else "") ,
                         on_click = SC.click_queue_refresh)
                
            with col2:
                st.button(":material/delete:", key="task-queue-clear", 
                          help = "Clear All Tasks" ,
                          on_click = SC.click_queue_clear_confirmation)
            
        if SC.queue_last_action:
            if SC.queue_last_action[1]:
                st.success(SC.queue_last_action[0] , icon = ":material/check_circle:")
            else:
                st.error(SC.queue_last_action[0] , icon = ":material/error:")
         
        if SC.task_queue.empty():
            st.info("Queue is empty, click the script below to run and it will be displayed here" , icon = ":material/queue_play_next:")
            return

        st.caption(f":rainbow[:material/bar_chart:] {SC.task_queue.status_message()}")
        st.markdown("")
        # show queue filters

        with st.container(key="queue-filter-container").expander("Queue Filters" , expanded = False , icon = ":material/filter_list:"):
            status_options = ["All" , "Running" , "Complete" , "Error"]
            folder_options = [item.path for item in SC.path_items if item.is_dir]
            file_options = [item.path for item in SC.path_items if item.is_file]
            st.radio(":gray-badge[**Running Status**]" , status_options , key = "queue-filter-status", horizontal = True ,
                     on_change = SC.click_queue_filter_status)
            st.multiselect(":gray-badge[**Script Folder**]" , folder_options , key = "queue-filter-path-folder" ,
                           format_func = lambda x: str(x.relative_to(BASE_DIR)) ,
                           on_change = SC.click_queue_filter_path_folder)
            st.multiselect(":gray-badge[**Script File**]" , file_options , key = "queue-filter-path-file" ,
                           format_func = lambda x: x.name ,
                           on_change = SC.click_queue_filter_path_file)
            
        queue = SC.filter_task_queue()
        with st.container():
            for item in queue.values():
                placeholder = st.empty()
                container = placeholder.container(key = f"queue-item-container-{item.id}")
                with container:
                    cols = st.columns([12, 1 , 1] , gap = "small" , vertical_alignment = "center")
                        
                    help_text = '|'.join([f"Status: {item.status}" , f"Dur: {item.duration_str}", f"PID: {item.pid}"])
                    cols[0].button(f"{item.tag_icon} {item.button_str}",  help=help_text , key=f"queue-item-content-{item.id}" , 
                                use_container_width=True , on_click = SC.click_queue_item , args = (item,))
                    
                    cols[1].button(":material/cancel:", 
                                   key=f"queue-item-remover-{item.id}", help="Remove/Terminate", type="tertiary",
                                   on_click = SC.click_queue_remove_item , args = (item,))
                    
                    cols[2].button(
                        ":material/slideshow:", 
                        key=f"show-complete-report-{'success' if item.status == 'complete' else 'error'}-{item.id}" ,
                        help = "Show complete report in main page" ,
                        on_click = SC.click_show_complete_report , args = (item,) , type="tertiary")
                    
                    if SC.running_report_queue is None or SC.running_report_queue != item.id:
                        continue
                
                    status_text = f'Running Report {item.status_state.title()}'
                    status = st.status(status_text , state = item.status_state , expanded = True)

                    with status:
                        col_config = {
                            'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                            'Value': st.column_config.TextColumn(width="large", help='Value of the item')
                        }

                        st.dataframe(item.dataframe() , row_height = 20 , column_config = col_config)
                        SC.wait_for_complete(item)
                        if item.status == 'complete':
                            st.success(f'Script Completed' , icon = ":material/add_task:")
                        elif item.status == 'error':
                            st.error(f'Script Failed' , icon = ":material/error:")
                        
def show_developer_info():
    """show developer info"""
    container = st.container(key = "developer-info-special-expander")
    with container.expander("**Developer Info**" , expanded = False , icon = ":material/bug_report:"):
        with st.expander("Session State" , expanded = False , icon = ":material/star:").container(height = 500):
                st.write(st.session_state)
        
        with st.expander("Session Control" , expanded = False , icon = ":material/settings:").container(height = 500):
                st.write(SC)

        SC.task_queue.refresh()     
        with st.expander("View queue file", expanded=False , icon = ":material/file_json:").container(height = 500):
            try:
                st.json(SC.task_queue.full_queue_dict() , expanded = 1)
            except Exception as e:
                st.error(f"Error loading queue: {e}")

        with st.expander("View action log", expanded=False , icon = ":material/format_list_numbered:"):
            st.code(ActionLogger.get_action_log(), language='log' , height = 500)

        with st.expander("View error log", expanded=False , icon = ":material/error:"):
            st.code(ActionLogger.get_error_log(), language='log' , height = 500)

        cols = st.columns(5) # 5 buttons in a row
        with cols[0]:
            st.button("Queue" , icon = ":material/directory_sync:" , key = "queue-refresh-developer" , 
                      help = "Refresh Queue" ,
                      on_click = SC.click_queue_refresh)
        with cols[1]:
            st.button("Queue" , icon = ":material/delete:" , key = "queue-clear-developer" , 
                      help = "Clear All Tasks in Queue" ,
                      on_click = SC.click_queue_clear_confirmation)
        with cols[2]:
            st.button("Log" , icon = ":material/delete:" , key = "clear-log-developer" , 
                      help = "Clear Both Action and Error Logs" ,
                      on_click = SC.click_log_clear_confirmation)
    
def show_config_editor():
    """show config yaml editor"""
    config_dir = BASE_DIR.parent.joinpath("configs")
    files = [f for sub in ["train" , "trade" , "nn" , "boost"] for f in config_dir.joinpath(sub).glob("*.yaml")]
    default_file = config_dir.joinpath("train/model.yaml")

    container = st.container(key="special-expander-editor")
    with container.expander("**YAML Editor**", expanded=False, icon=":material/edit_document:"):
        st.info(f"This File Editor is for editing selected config files", icon=":material/info:")
        st.info(f"For other config files, please use the file explorer", icon=":material/info:")
        
        config_editor = YAMLFileEditor('config-editor', file_root=config_dir)
        SC.config_editor_state = config_editor.state
        config_editor.show_yaml_editor(files, default_file=default_file)
    
def show_folder():
    """show folder content recursively"""
    items = SC.path_items
    for item in items:
        if item.is_dir:
            folder_name = re.sub(r'^\d+_', '', item.name).replace('_', ' ').title()
            body = f"""
            <div style="
                font-size: 20px;
                font-weight: bold;
                margin-top: 5px;
                margin-bottom: 5px;
                margin-left: {(item.level)*15}px;
            ">📂 {folder_name}</div>
            """       
            st.markdown(body , unsafe_allow_html=True)
 
        elif item.level > 0:
            show_script_runner(item.script_runner())

def show_script_runner(runner: ScriptRunner):
    """show single script runner"""
    SC.script_runners[runner.script_key] = runner
    with st.container(key = f"script-container-level-{runner.level}-{runner.script_key}"):
        button_text = ':no_entry:' if runner.header.disabled else ':snake:' + ' ' + runner.desc
        selected = SC.current_script_runner is not None and SC.current_script_runner == runner.script_key
        widget_key = f"script-runner-expand-{runner.script_key}" if not selected else f"script-runner-expand-selected-{runner.script_key}"
        st.button(f"**{button_text}**" , key=widget_key , 
                    help = f":material/info: **{runner.content}** \n*{str(runner.script)}*" ,
                    on_click = SC.click_script_runner_expand , args = (runner,))

def show_script_details(runner : ScriptRunner | None):
    """show script details"""
    if runner is None:
        return
    if todo := runner.header.todo:
        st.info(f":material/pending_actions: {todo}")
    # st.button("Filter Script Tasks", icon = ":material/filter_list:", 
    #          key = f"script-runner-filter-{runner.script_key}", 
    #          help = f":material/info: **Click to filter all tasks of this script**" ,
    #          on_click = SC.click_script_runner_filter , args = (runner,))
    queue = SC.task_queue.filter(file = [runner.path.path])
    if queue:
        expander = st.expander("Choose Task Item from Queue", 
                               expanded = False , #SC.detail_item_expander_expanded , 
                               icon = ":material/checklist:")
        with expander:
            for item in SC.task_queue.filter(file = [runner.path.path]).values():
                col0 , col1 = expander.columns([14, 1] , gap = "small" , vertical_alignment = "center")
                with col0:
                    button_key = f"choose-item-select-{item.id}" if SC.choose_task_item != item.id else f"choose-item-selected-{item.id}"
                    info_text = f"--ID {item.time_id} --Status {item.status.title()} --Dur {item.duration_str}"
                    st.button(f"{item.icon} {item.button_str} {info_text}", 
                              key=button_key , 
                              use_container_width=True , on_click = SC.click_item_choose_select , args = (item,))
                with col1.container(key = f"choose-item-remover-{item.id}"):
                    st.button(":material/cancel:", key = f"choose-item-remover-button-{item.id}",
                              help="Remove/Terminate", type="tertiary" ,
                              on_click = SC.click_item_choose_remove , args = (item,))

    if SC.choose_task_item:
        st.success(f"Task Item {SC.choose_task_item} chosen" , icon = ":material/check_circle:")

    if runner.disabled:
        st.error(f":material/disabled_by_default: This script is disabled")
        return
    
    with st.container(key = f"script-setting-container-{runner.script_key}"):
        param_inputs = runner.header.get_param_inputs()
        settings_col , collapse_col = st.columns([1, 1] , vertical_alignment = "center")
        with settings_col:
            if not param_inputs:
                st.info("**No parameter settings**" , icon = ":material/settings:")
            else:
                st.info("**Parameter Settings**" , icon = ":material/settings:")

        with collapse_col:
            st.button(":material/close:", key=f"script-setting-classic-remover-{runner.script_key}", 
                      help="Collapse", type="secondary" ,
                      on_click = SC.click_script_runner_expand , args = (runner,))                
        
    params = ParamInputsForm(runner).init_param_inputs('customized').param_values
    if SC.ready_to_go(runner):
        help_text = f"Parameters valid, run {runner.script_key}"
        button_key = f"script-runner-run-enabled-{runner.script_key}"
    else:
        help_text = f"Parameters invalid, please check required ones"
        button_key = f"script-runner-run-disabled-{runner.script_key}"
    st.button(":material/mode_off_on:", key=button_key , 
            help = help_text , disabled = not SC.ready_to_go(runner) , 
            on_click = SC.click_script_runner_run , args = (runner,params))
    show_report_main(runner)

class ParamInputsForm:
    def __init__(self , runner : ScriptRunner):
        self.runner = runner
        self.param_list = [self.WidgetParamInput(runner, p) for p in runner.header.get_param_inputs()]
        self.errors = []

    def init_param_inputs(self , type : Literal['customized', 'form'] = 'customized' , 
                          trigger_item : TaskItem | None = None):
        trigger_item = SC.task_queue.get(SC.script_runner_trigger_item)
        cmd = trigger_item.cmd if trigger_item is not None else ''
        SC.script_runner_trigger_item = None
        if type == 'customized':
            self.init_customized_container(cmd)
        elif type == 'form':
            self.init_form(cmd)
        else:
            raise ValueError(f"Invalid param inputs type: {type}")
        return self

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
            if self.script_key not in SC.script_params_cache:
                SC.script_params_cache[self.script_key] = {
                    'raw': {},
                    'value': {},
                    'valid': {}
                }
            SC.script_params_cache[self.script_key]['raw'][self.name] = self.raw_value
            SC.script_params_cache[self.script_key]['value'][self.name] = self.param_value
            SC.script_params_cache[self.script_key]['valid'][self.name] = self.is_valid()

    def cmd_to_param_values(self , cmd : str = ''):
        param_values = {}
        if not cmd or not str(self.runner.path.path) in cmd: return param_values
        main_str = [s for s in cmd.split(";") if str(self.runner.path.path) in s][0]
        param_str = ''.join(main_str.split(str(self.runner.path.path))[1:]).strip()
        
        for param_str in param_str.split('--'):
            if not param_str: continue
            param_name , param_value = param_str.split(' ' , 1)
            value = param_value.strip()
            if value == 'True': value = True
            elif value == 'False': value = False
            elif value == 'None': value = None
            param_values[param_name] = value
        return param_values

    def init_customized_container(self , cmd : str = '' , num_cols : int = 3):
        num_cols = min(num_cols, len(self.param_list))
        cmd_param_values = self.cmd_to_param_values(cmd)
        for i, wp in enumerate(self.param_list):
            if i % num_cols == 0:
                param_cols = st.columns(num_cols)
            with param_cols[i % num_cols]:
                self.get_widget(
                    runner=self.runner, param=wp, value = cmd_param_values.get(wp.name) ,
                    on_change=self.on_widget_change, args=(wp,))
                self.on_widget_change(wp)
                if err_msg := wp.error_message():
                    st.error(err_msg , icon = ":material/error:")
        return self
    
    def init_form(self , cmd : str):
        cmd_param_values = self.cmd_to_param_values(cmd)
        with st.form(f"ParamInputsForm-{self.runner.script_key}" , clear_on_submit = False):
            for param in self.param_list:
                self.get_widget(self.runner, param, value = cmd_param_values.get(param.name))

            if st.form_submit_button(
                "Submit" ,
                help = "Submit Parameters to Run Script" ,
            ):
                self.submit()

        return self

    @property
    def param_values(self):
        return {wp.name: wp.param_value for wp in self.param_list}

    def validate(self):
        self.errors = []
        for wp in self.param_list:
            if err_msg := wp.error_message():
                self.errors.append(err_msg)
                
        return len(self.errors) == 0

    def submit(self):
        for wp in self.param_list: wp.on_change()
            
        if not self.validate():
            for err_msg in self.errors:
                st.error(err_msg , icon = ":material/error:")
    
    def process(self):
        ...

    @classmethod
    def get_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                   value : Any | None = None ,
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
        
        return func(runner, param, value, on_change, args, kwargs)

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
        default_value = SC.script_params_cache.get(runner.script_key, {}).get('raw', {}).get(param.name, param.default)
        if default_value is None:
            default_value = st.session_state[widget_key] if widget_key in st.session_state else param.default
        return default_value
    
    @classmethod
    def get_widget_value(cls , runner : ScriptRunner , param : ScriptParamInput):
        widget_key = cls.get_widget_key(runner, param)
        return st.session_state[widget_key]

    @classmethod
    def list_widget(cls , runner : ScriptRunner , param : ScriptParamInput , 
                    value : Any | None = None ,
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert isinstance(ptype, list) , f"Param {param.name} is not a list"
        
        widget_key = cls.get_widget_key(runner, param)
        default_value = str(value) if value is not None else cls.get_default_value(runner, param)
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
                    value : Any | None = None ,
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == str , f"Param {param.name} is not a string"
        widget_key = cls.get_widget_key(runner, param)
        default_value = str(value) if value is not None else cls.get_default_value(runner, param)
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
                    value : Any | None = None ,
                    on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == bool , f"Param {param.name} is not a boolean"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = value if value is not None else cls.get_default_value(runner, param)
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
                   value : Any | None = None ,
                   on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == int , f"Param {param.name} is not an integer"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = value if value is not None else cls.get_default_value(runner, param)
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
                     value : Any | None = None ,
                     on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        assert ptype == float , f"Param {param.name} is not a float"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = value if value is not None else cls.get_default_value(runner, param)
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
    item = SC.task_queue.get(SC.current_task_item)
    if item is None: return
    if not item.belong_to(runner): return

    status_text = f'Running Report {item.status_state.title()}'
    status_placeholder = st.empty()
    status = status_placeholder.status(status_text , state = item.status_state , expanded = True)
       
    with status_placeholder:
        if not SC.running_report_main_cleared:
            st.write('')
            SC.running_report_main_cleared = True
            SC.running_report_file_previewer = None
            st.rerun()

        with status:
            with st.expander(":rainbow[:material/build:] **Command Details**", expanded=False):
                st.code(item.cmd , wrap_lines=True)

            script_str = f"Script [{item.format_path}] ({item.time_str()}) (PID: {item.pid})"
            st.success(f'{script_str} started' , icon = ":material/add_task:")

            df_placeholder = st.empty()
            col_config = {
                'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                'Value': st.column_config.TextColumn(width="large", help='Value of the item')
            }
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(info_type = 'enter') , row_height = 20 , column_config = col_config)

            SC.wait_for_complete(item)
            SC.task_queue.refresh()
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(info_type = 'enter') , row_height = 20 , column_config = col_config)

            if item.status == 'error':
                st.error(f'{script_str} has error' , icon = ":material/error:")
            else:
                st.success(f'{script_str} Completed' , icon = ":material/trophy:")

            exit_info_list = item.info_list(info_type = 'exit')
            
            with st.expander(f":rainbow[:material/fact_check:] **Exit Information**", expanded=True):
                for name , value in exit_info_list:
                    st.badge(f"**{name}**" , color = "blue")
                    for s in value.split('\n'):
                        st.write(ColoredText(s))
                    st.markdown('')

            if item.exit_files:
                with st.expander(f":rainbow[:material/file_present:] **File Previewer**", expanded=True):
                    for file in item.exit_files:
                        path = Path(file).absolute()
                        preview_key = f"file-preview-{path}"
                        col1, col2 = st.columns([4, 1] , vertical_alignment = "center")
                        with col1:
                            st.button(path.name, key=preview_key , icon = ":material/file_present:" , 
                                      help = f"Preview {path}" ,
                                      on_click = SC.click_file_preview , args = (path,))

                        with col2.container(key = f"file-download-{path}"):
                            with open(file_path, 'rb') as f:
                                if st.download_button(
                                    ':material/download:', 
                                    data=f.read(),
                                    file_name=str(path),
                                    key = f"download-{path}",
                                    help = f"Download {path}",
                                    on_click=SC.click_file_download , args = (path,)
                                ):
                                    pass

                    previewer = FilePreviewer(SC.running_report_file_previewer)
                    previewer.preview()

def show_main_part():
    """show main part"""
    col_folder , col_details = st.columns([1, 3] , gap = 'small')
    with col_folder.container():
        header_container = st.container(height = 90)
        content_container = st.container(height = 750)
        with header_container:
            st.subheader(":blue[:material/folder: **Script Folder**]")
            no_wrap_info("Click Script to Expand")
        with content_container:
            show_folder()   
    with col_details:
        header_container = st.container(height = 90)
        content_container = st.container(height = 750)
        runner = SC.script_runners[SC.current_script_runner] if SC.current_script_runner else None
        with header_container:
            if runner is None:
                st.subheader(":blue[:material/code: **No Script Selected**]")
            else:
                st.subheader(f":blue[:material/code: **{runner.script_key}**]")
                no_wrap_info(runner.content)   
        with content_container:
            show_script_details(runner)

def no_wrap_info(text : str , color : str = 'gray' , icon : str = '💬'):
    """no wrap info"""
    text = f"{icon} {text}" if text else ''
    content = f"""<p style="font-size: 16px;
                color: {color};
                padding-top: 10px;
                padding-bottom: 10px;
                border-radius: 10px;
                white-space: nowrap;
                text-decoration: underline double gray 1px;
                ">
                {text}
                </p>
                """
    st.markdown(content, unsafe_allow_html = True)

def main():
    page_config()
    page_css()
    show_sidebar()
    show_developer_info()
    show_config_editor()
    show_main_part()
    
if __name__ == '__main__':
    main() 