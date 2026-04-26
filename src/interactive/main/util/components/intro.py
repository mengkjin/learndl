import streamlit as st
import platform
import psutil
import torch
import sys
import re

from src.proj import Const , PATH
from src.interactive.backend import ScriptRunner
from src.interactive.frontend import subheader_expander , YAMLFileEditor
from src.interactive.main.util.session_control import SC


__all__ = [
    'show_tutorial' , 'show_system_info' , 'show_pending_features' , 'show_config_editor' , 
    'show_developer_info' , 'show_script_structure']

def show_tutorial() -> None:
    """Render the tutorial expander with numbered step instructions."""
    with subheader_expander('Tutorial' , ':material/school:' , True , help = 'Basic Tutorial for the Project.' , key = 'home-tutorial'):
        st.markdown("""
        1. :material/settings: Click the script button to expand the parameter settings
        2. :green[:material/mode_off_on:] Fill in the necessary parameters and click Run
        3. :blue[:material/bar_chart:] View the running report and generated files
        4. :gray[:material/file_present:] Preview the generated HTML/PDF files
        """)

def show_system_info() -> None:
    """Render the system info expander (OS, memory, GPU, CPU, Python, Streamlit)."""
    options : dict[str, str] = {}
    # os
    options[':material/keyboard_command_key: OS'] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    # memory
    mem = psutil.virtual_memory()
    options[':material/memory: Memory Usage'] = \
        f"{(mem.total - mem.available) / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB ({mem.percent:.1f}%)"
    # gpu
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = "**GPU Usage (CUDA)**" , f"{used:.1f} / {total:.1f} GB ({used / total * 100:.1f}%)"
    elif torch.backends.mps.is_available():
        used = torch.mps.current_allocated_memory() / 1024**3
        if torch.__version__ >= '2.3.0':
            recommend = torch.mps.recommended_max_memory() / 1024**3 # type:ignore
            gpu_info = "GPU Usage (MPS)" , f"{used:.1f} / {recommend:.1f} GB ({used / recommend * 100:.1f}%)"
        else:
            gpu_info = "GPU Usage (MPS)" , f"{used:.1f} GB Used"
    else:
        gpu_info = "GPU Usage (None)" , "No GPU"
    options[f':material/memory_alt: {gpu_info[0]}'] = f"{gpu_info[1]}"
    # cpu
    options[':material/select_all: CPU Usage'] = f"{psutil.cpu_percent():.1f}%"
    # python
    options[':material/commit: Python Version'] = f"{sys.version.split(' ')[0]}"
    # streamlit
    options[':material/commit: Streamlit Version'] = f"{st.__version__}"
    
    with subheader_expander('System Info' , ':material/computer:' , True , help = 'System Info , includes OS, memory, GPU, CPU, Python, and Streamlit version.' , key = 'home-system-info'):
        for i , (label , value) in enumerate(options.items()):
            cols = st.columns([2,3])
            cols[0].markdown(f"***{label}***")
            cols[1].markdown(f":blue-badge[*{value}*]")
        
def show_pending_features() -> None:
    """Render warning badges for any pending features configured in preferences."""
    if not (pending_features := Const.Pref.interactive.get('pending_features' , [])):
        return
    with subheader_expander('Pending Features' , ':material/pending_actions:' , True , key = 'home-pending-features'):
        for feature in pending_features:
            st.warning(feature , icon = ":material/schedule:")

def show_config_editor() -> None:
    """Render the YAML file editor scoped to model and algo config files.

    Discovers all ``*.yaml`` files under ``configs/model/`` and
    ``configs/algo/``, pre-selects ``configs/model/model.yaml``, and passes
    them to a :class:`YAMLFileEditor` widget.
    """
    with st.container(key="config-editor-container"):
        files = [f for sub in ["model" , "algo"] for f in PATH.conf.joinpath(sub).rglob("*.yaml")]
        default_file = PATH.conf.joinpath("model" , "model.yaml")
        config_editor = YAMLFileEditor('config-editor', file_root=PATH.conf)
        SC.config_editor_state = config_editor.state
        config_editor.show_yaml_editor(files, default_file=default_file)

def show_developer_info(H = 500):
    """show developer info"""
    segments = {
        "Session Control" : {
            'icon' : ':material/settings:' ,
            'operation' : lambda : st.write(SC) 
        } , 
        "Session States"  : {
            'icon' : ':material/star:' ,
            'operation' : lambda : st.write(st.session_state) 
        } , 
        "Task Queue" : {
            'icon' : ':material/directory_sync:' ,
            'operation' : lambda : st.json(SC.task_queue.queue_content() , expanded = 1)
        } , 
    }

    def developer_info_selected_change() -> None:
        """Callback: handle ``'All'`` / ``'None'`` shortcuts in the multi-select control."""
        selected = getattr(st.session_state , 'developer-info-selected' , [])
        if 'All' in selected:
            st.session_state['developer-info-selected'] = ['Session Control' , 'Session States' , 'Task Queue'] # , 'Action Logs' , 'Error Logs']
        if 'None' in selected:
            st.session_state['developer-info-selected'] = []
    
    with st.container(key = "developer-info-container"):
        col_name , col_widget = st.columns([1,3])
        col_name.info('Select Developer Info Types')
        col_widget.segmented_control('developer-info-selected' , 
                                options = ["All" , "None"] + list(segments.keys()) , 
                                key = "developer-info-selected" , 
                                selection_mode = "multi" ,
                                default = list(segments.keys()) , label_visibility = "collapsed" ,
                                on_change = developer_info_selected_change)
        
        col_name , col_widget = st.columns([1,3])
        col_name.info('Developer Level Operations')
        with col_widget.container(key = "developer-info-buttons"):
            cols = st.columns(5)
            cols[0].button("Log" , icon = ":material/delete_forever:" , key = "developer-log-clear" , 
                        help = "Clear Both Action and Error Logs" ,
                        on_click = SC.click_log_clear_confirmation)
        
        for seg , content in segments.items():
            if seg not in st.session_state['developer-info-selected']: 
                continue
            with subheader_expander(f'developer-info-{seg}' , seg , content['icon'] , height = H):
                content['operation']()

def show_script_structure():
    """show folder content recursively"""  
    def show_script_runner(runner: ScriptRunner):
        """show single script runner"""
        if runner.script_key not in SC.script_runners: 
            SC.script_runners[runner.script_key] = runner
        
        page = SC.get_page(runner.script_key)
        if page is None: 
            return
        
        with st.container(key = f"script-structure-level-{runner.level}-{runner.script_key}"):
            cols = st.columns([1, 1] , gap = "small" , vertical_alignment = "center")
            
            with cols[0]:
                button_text = ':no_entry:' if runner.header.disabled else ':snake:' + ' ' + runner.desc
                widget_key = f"script-runner-expand-{runner.script_key}"
                if st.button(f"**{button_text}**" , key=widget_key , 
                            help = f"*{str(runner.script)}*"):
                    st.switch_page(page['page'])
            with cols[1]:
                st.info(f"**{runner.content}**" , icon = ":material/info:")
    with st.container(key="script-structure-container"):
        items = SC.path_items
        for item in items:
            if item.is_dir:
                folder_name = re.sub(r'^\d+_', '', item.name).replace('_', ' ').title()
                body = f"""
                <div style="
                    font-size: 18px;
                    font-weight: bold;
                    margin-top: 5px;
                    margin-bottom: 5px;
                    letter-spacing: 3px;
                    margin-left: {(item.level)*45}px;
                ">📂 {folder_name}</div>
                """       
                st.markdown(body , unsafe_allow_html=True)

            elif item.level > 0:
                show_script_runner(item.script_runner())
