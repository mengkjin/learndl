'''use streamlit to run scripts'''

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

import os, platform, subprocess, yaml, re, time, base64, glob, json, signal
import streamlit as st
import streamlit.components.v1 as components
import psutil
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from src_runs.util import terminal_cmd
from src_runs.util.exception import OutOfRange , Unspecified

def st_config():
    """config streamlit page and add compact CSS style"""
    st.set_page_config(
        page_title="Script Runner",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # add compact CSS style
    st.markdown("""
    <style>
    .stButton > button {
        height: 18px;
        font-size: 12px;
        display: flex;
        min-height: 18px;
        line-height: 1;
        justify-content: left;
        text-align: center;
        align-items: center;
        justify-items: center;
        justify-self: center;
        margin: 0 !important;
        padding: 0px !important;
    }
    .stSelectbox > div > div {
        height: 14px;
        font-size: 12px;
    }
    .stTextInput > div > div > input {
        height: 14px;
        font-size: 12px;
    }
    .stNumberInput > div > div > input {
        height: 14px;
        font-size: 12px;
    }
    .element-container {
        margin-bottom: 0px;
        display: flex;
        align-items: center;
    }
    .stMarkdown {
        padding-left: 0px;
        padding-right: 0px;
        padding-top: 0px;
        padding-bottom: 0px;
        margin-bottom: 1px;
        line-height: 1.1;
        display: flex;
        align-items: center;
    }
    .stMarkdown p {
        margin-top: 0px;
        margin-bottom: 1px;
        line-height: 1.1;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 4px;
        border-radius: 3px;
        margin: 0px;
    }
    .stContainer {
        border-radius: 0px;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        margin: 0px !important;
        padding: 0px !important;        
    }
    div[data-testid="column"] {
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sidebar():
    """streamlit sidebar for script runner"""
    with st.sidebar:
        st.markdown("### 📋 Usage")
        st.markdown("""
        1. 📁 folder is automatically expanded, no need to click
        2. ▶️ click script button to expand parameter settings
        3. 📝 fill necessary parameters and click run
        4. 📊 view run report and generated files
        5. 👁️ preview generated HTML/PDF files
        """)
        
        st.markdown("### 🔧 System Info")
        st.info(f"**OS:** {platform.system()}")
        st.info(f"**Memory:** {psutil.virtual_memory().percent:.1f}%")
        st.info(f"**CPU:** {psutil.cpu_percent():.1f}%")
        
        st.markdown("### 🎯 New Features")
        st.success("✅ folder is automatically expanded, no need to click")
        st.success("✅ compact interface design") 
        st.success("✅ script exclusive expand")
        st.success("✅ run report generation")
        st.success("✅ file preview function")
        
        st.markdown("### 🐛 Debug Info")
        if st.button("View Queue File", key="debug_queue"):
            queue_file = "run_queue.json"
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    queue_content = f.read()
                st.code(queue_content, language='json')
            else:
                st.info("queue file not found")

def create_main_content():
    """streamlit main content for script runner"""
    st.title("🚀 Project Script Runner")
    create_process_queue()
    create_main_folder()
    
def create_process_queue():
    """streamlit process queue for script runner"""
    queue = ProcessQueue()
    queue.show()
    st.markdown("---")

def create_main_folder():
    """streamlit main folder for script runner"""
    show_folder(Path("src_runs"))

def show_folder(folder_path: Path | str, level: int = 0, ignore_scripts: list[str] = ['widget.py' , 'streamlit.py'] , 
                max_level: int = 3):
    """recursively show folder content"""
    if not Path(folder_path).exists():
        st.error(f"folder not found: {folder_path}")
        return
    
    if level >= max_level:
        return
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        st.error(f"folder not found: {folder_path}")
        return
        
    # get all items and sort
    items = [item for item in folder_path.iterdir() if not item.name.startswith(('.', '_')) and item.name not in ignore_scripts]
    items.sort(key=lambda x: (x.is_file(), x.name))
    
    # show folder title (more compact style)
    if level > 0:
        folder_name = folder_path.name.replace('_', ' ').title()
        st.markdown(f"**📁 {folder_name}**")
    
    # handle subfolders and files
    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file() and item.suffix == '.py']
    
    # show subfolders (default expanded)
    for folder in folders:
        show_folder(folder, level + 1 , max_level=max_level)
    
    # show Python scripts (compact mode)
    for script_file in files:
        ScriptRunner(script_file).show()

def run_script(script: str | Path, close_after_run=False, **kwargs):
    cmd = terminal_cmd(script, kwargs, close_after_run=close_after_run)
    script_name = Path(script).stem
    
    # add to queue
    queue_item = add_to_queue(script_name, cmd)
    st.info(f"✅ added to queue: {queue_item['id']}")
    
    try:
        # start process
        process = subprocess.Popen(cmd, shell=True, encoding='utf-8')
        
        # update queue status
        update_queue_item(queue_item['id'], {
            'pid': process.pid,
            'status': 'running',
            'start_time': time.time()
        })
        
        st.success(f'✅ Run Script Started! PID: {process.pid}')
        st.info('📊 please click the "🔄 refresh" button in the queue area to see the latest status')
        
        # show command info
        with st.expander("🔧 Command Info", expanded=False):
            st.code(cmd)
        
    except Exception as e:
        # update queue status to failed
        update_queue_item(queue_item['id'], {
            'status': 'failed',
            'error': str(e),
            'end_time': time.time()
        })
        st.error(f'❌ Run Script Failed: {str(e)}')


def load_output_manifest(script_name):
    """load script output file manifest"""
    manifest_file = "output_manifest.json"
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, 'r') as f:
                data = json.load(f)
                if data.get('script') == script_name:
                    return data.get('files', [])
        except:
            pass
    return []

def show_run_report(queue_item):
    """show run report"""
    if queue_item['status'] != 'completed':
        st.warning("script not completed, cannot generate complete report")
        return
    
    duration = queue_item.get('end_time', 0) - queue_item.get('start_time', 0)
    
    st.subheader("📊 Script Run Report")
    
    # basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Run Time", f"{duration:.2f} seconds")
    with col2:
        st.metric("Script Name", queue_item['script_name'])
    with col3:
        start_time_str = datetime.fromtimestamp(queue_item['start_time']).strftime('%H:%M:%S')
        st.metric("Start Time", start_time_str)
    
    # check script output files
    output_files = load_output_manifest(queue_item['script_name'])
    
    # show generated files
    if output_files:
        st.subheader("📁 Generated Files")
        for file_path in output_files:
            if os.path.exists(file_path):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file_path}")
                with col2:
                    if file_path.endswith('.html'):
                        if st.button("Preview", key=f"preview_{file_path}_{queue_item['id']}"):
                            preview_html_file(file_path)
                    elif file_path.endswith('.pdf'):
                        if st.button("Preview", key=f"preview_{file_path}_{queue_item['id']}"):
                            preview_pdf_file(file_path)
                with col3:
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                "Download", 
                                data=f.read(),
                                file_name=os.path.basename(file_path),
                                key=f"download_{file_path}_{queue_item['id']}"
                            )
                    except:
                        st.error("file read failed")
            else:
                st.warning(f"⚠️ file not found: {file_path}")
    else:
        st.info("💡 **Tip**: script can report generated files by creating `output_manifest.json` file")
        with st.expander("📖 How to output file manifest in script", expanded=False):
            st.code('''
import json
from datetime import datetime

# add the following code before script ends
output_files = ["output1.html", "output2.pdf"]  # your output files list

manifest = {
    "script": "your_script_name",
    "files": output_files,
    "timestamp": datetime.now().isoformat()
}

with open("output_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
            ''', language='python')

def preview_html_file(file_path):
    """preview HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.subheader(f"📄 {os.path.basename(file_path)}")
        components.html(html_content, height=600, scrolling=True)
    except Exception as e:
        st.error(f"cannot preview HTML file: {str(e)}")

def preview_pdf_file(file_path):
    """preview PDF file"""
    try:
        with open(file_path, 'rb') as f:
            pdf_data = f.read()
        
        # 使用base64编码PDF
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{pdf_base64}" 
                width="100%" height="600px" type="application/pdf">
        </iframe>
        '''
        st.subheader(f"📄 {os.path.basename(file_path)}")
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"cannot preview PDF file: {str(e)}")
        st.info("you can download the file and view it")

class ScriptRunner:
    def __init__(self, script_path: Path | str):
        self.script = Path(script_path).absolute()
        self.header = self.parse_script_header()
        self.script_key = str(self.script.absolute())
        
    def parse_script_header(self, verbose=False, include_starter='#', exit_starter='', 
                          ignore_starters=('#!', '# coding:')):
        header_dict = {}
        yaml_lines: list[str] = []
        
        try:
            with open(self.script, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith(ignore_starters): 
                        continue
                    elif stripped_line.startswith(include_starter):
                        yaml_lines.append(stripped_line)
                    elif stripped_line.startswith(exit_starter):
                        break

            yaml_str = '\n'.join(line.removeprefix(include_starter) for line in yaml_lines)
            header_dict = yaml.safe_load(yaml_str) or {}
            
        except FileNotFoundError:
            header_dict = {
                'disabled': True,
                'description': 'file not found',
                'content': f'file not found: {self.script}'
            }
        except yaml.YAMLError as e:
            header_dict = {
                'disabled': True,
                'description': 'YAML parse error',
                'content': f'error info: {e}'
            }
        except Exception as e:
            header_dict = {
                'disabled': True,
                'description': 'file read error',
                'content': f'error info: {e}'
            }

        if 'description' not in header_dict:
            header_dict['description'] = self.script.name
            
        return header_dict

    def get_param_inputs(self):
        """generate parameter input widgets and return parameter values"""
        param_inputs = self.header.get('param_inputs', {})
        if not param_inputs:
            return {}
            
        st.subheader("Parameter Settings")
        params = {}
        
        # create 3 columns - all parameters displayed at once
        param_items = list(param_inputs.items())
        num_cols = min(3, len(param_items))
        param_cols = st.columns(num_cols)
        
        # collect all parameters first, avoid gradual display due to dependencies
        all_widgets = []
        for i, (pname, pdef) in enumerate(param_items):
            col_idx = i % num_cols
            all_widgets.append((col_idx, pname, pdef))
        
        # render all parameters at once
        for col_idx, pname, pdef in all_widgets:
            with param_cols[col_idx]:
                try:
                    # parse parameter definition
                    ptype = pdef.get("type") or pdef.get("enum")
                    if isinstance(ptype, str):
                        ptype = eval(ptype)
                    elif isinstance(ptype, (list, tuple)):
                        ptype = list(ptype)
                        
                    required = pdef.get('required', False)
                    default = pdef.get('default')
                    desc = pdef.get('desc', pname)
                    prefix = pdef.get('prefix', '')
                    
                    # generate unique key
                    key = f"{self.script.name}_{pname}"
                    
                    # create input widget
                    if isinstance(ptype, list):
                        # dropdown select
                        options = [f'{prefix}{e}' for e in ptype]
                        placeholder = f"select {desc}" if required else f"optional: {desc}"
                        
                        if default is not None:
                            default_idx = ptype.index(default) if default in ptype else 0
                            value = st.selectbox(
                                f"**{desc}**" if required else desc,
                                options,
                                index=default_idx,
                                key=key
                            )
                        else:
                            value = st.selectbox(
                                f"**{desc}**" if required else desc,
                                [placeholder] + options,
                                key=key
                            )
                        
                        # handle select result
                        if value == placeholder:
                            if required:
                                st.error(f"Select a valid value for [{desc}]")
                                return None
                            else:
                                params[pname] = None
                        else:
                            # remove prefix and find original value
                            enum_idx = options.index(value)
                            params[pname] = ptype[enum_idx]
                            
                    elif ptype == bool:
                        # boolean switch
                        default_val = bool(eval(str(default))) if default is not None else False
                        value = st.toggle(
                            f"**{desc}**" if required else desc,
                            value=default_val,
                            key=key
                        )
                        params[pname] = value
                        
                    elif ptype in [str, int, float]:
                        # text/number input
                        placeholder = f"input {desc}" if required else f"optional: {desc}"
                        
                        if ptype == str:
                            value = st.text_input(
                                f"**{desc}**" if required else desc,
                                value=str(default) if default is not None else "",
                                placeholder=placeholder,
                                key=key
                            )
                            if required and (not value or value.strip() == ""):
                                st.error(f"please input a valid value for [{desc}]")
                                return None
                            params[pname] = value if value.strip() else None
                            
                        elif ptype in [int, float]:
                            min_val = pdef.get('min')
                            max_val = pdef.get('max')
                            
                            if ptype == int:
                                value = st.number_input(
                                    f"**{desc}**" if required else desc,
                                    value=int(default) if default is not None else (min_val or 0),
                                    min_value=min_val,
                                    max_value=max_val,
                                    step=1,
                                    key=key
                                )
                            else:  # float
                                value = st.number_input(
                                    f"**{desc}**" if required else desc,
                                    value=float(default) if default is not None else (min_val or 0.0),
                                    min_value=min_val,
                                    max_value=max_val,
                                    step=0.1,
                                    key=key
                                )
                            params[pname] = value
                            
                except Exception as e:
                    st.error(f"parameter [{pname}] config error: {str(e)}")
                    return None
                
        return params

    def render(self):
        """render script interface (keep compatibility)"""
        self.render_details()
    
    def render_details(self):
        """render script details interface"""
        # TODO info
        if todo := self.header.get('TODO'):
            st.info(f"📝 TODO: {todo}")
            
        # check if disabled
        if self.header.get('disabled', False):
            st.error("script is disabled")
            return
            
        # get parameter inputs
        params = self.get_param_inputs()
        if params is None:  # parameter validation failed
            return
            
        # run button
        if st.button("🚀 运行脚本", key=f"run_{self.script.name}", type="primary", use_container_width=True):
            # add default parameters
            run_params = {
                'email': int(self.header.get('email', 0)),
                'close_after_run': bool(self.header.get('close_after_run', False))
            }
            run_params.update({k: v for k, v in params.items() if v is not None})
            
            # run script
            run_script(self.script, **run_params)

    def show(self):
        """show single script (compact mode, support exclusive expand)"""
        
        # initialize session state
        if 'current_script' not in st.session_state:
            st.session_state.current_script = None
        
        # script title row (compact layout, vertical center)
        col1, col2 = st.columns([5, 1])
        with col1:
            script_name = self.script.stem.replace('_', ' ').title()
            
            # check if current script is expanded
            is_current = st.session_state.current_script == self.script_key
            button_text = f"{'🔽' if is_current else '▶️'} 🐍 {script_name}"
            
            # use horizontal layout, ensure vertical center
            button_col, desc_col = st.columns([2, 3])
            with button_col:
                if st.button(button_text, key=f"toggle_{self.script_key}" , use_container_width=True):
                    # toggle script expand state
                    if st.session_state.current_script == self.script_key:
                        st.session_state.current_script = None
                    else:
                        st.session_state.current_sript = self.script_key
                        st.rerun()
                
            with desc_col:
                content = self.header.get('content', '')
                st.markdown(f'<div style="display: flex; align-items: center; justify-items: center; height: 14px; font-size: 11px; color: #666;">'
                            f'💬 {content[:80]}{"..." if len(content) > 80 else ""}</div>', 
                            unsafe_allow_html=True)
        
        with col2:
            # show script status
            if self.header.get('disabled', False):
                st.markdown("🚫")
            else:
                st.markdown("✅")
        
        # only show detailed content for current script
        if st.session_state.current_script == self.script_key:
            st.markdown("---")
            self.render_details()


class ProcessQueue:
    """streamlit process queue for script runner"""
    def __init__(self):
        self.queue = load_queue()

    def __call__(self):
        return self.queue
    
    def show(self):
        """show run queue"""
        # queue title and refresh button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("🔄 Run Queue")
        with col2:
            if st.button("🔄 Refresh", key="refresh_queue"):
                st.rerun()
        
        if not self.queue:
            st.info("Empty queue, click the script below to run and show progress here")
            return
        
        # update queue status
        updated_queue = []
        status_changed = False
        
        for item in self.queue:
            if item.get('pid') and item['status'] == 'running':
                if check_process_status(item['pid']):
                    # process is still running
                    item['status'] = 'running'
                else:
                    # process has ended
                    old_status = item['status']
                    item['status'] = 'completed'
                    item['end_time'] = time.time()
                    if old_status != 'completed':
                        status_changed = True
            updated_queue.append(item)
        
        if status_changed:
            save_queue(updated_queue)
            queue = updated_queue
        
        # show queue stats
        running_count = len([item for item in self.queue if item['status'] == 'running'])
        completed_count = len([item for item in self.queue if item['status'] == 'completed'])
        failed_count = len([item for item in self.queue if item['status'] == 'failed'])
        
        if running_count > 0 or completed_count > 0 or failed_count > 0:
            st.caption(f"📊 Running: {running_count} | Completed: {completed_count} | Failed: {failed_count}")
        
        # show queue items
        for item in self.queue:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    status_icon = {
                        'starting': '🟡',
                        'running': '🟢', 
                        'completed': '✅',
                        'failed': '❌'
                    }.get(item['status'], '❓')
                    
                    st.markdown(f"**{status_icon} {item['script_name']}**")
                    
                with col2:
                    if item['status'] == 'running' and item.get('start_time'):
                        duration = time.time() - item['start_time']
                        st.write(f"⏱️ {duration:.0f} seconds")
                    elif item['status'] == 'completed' and item.get('end_time') and item.get('start_time'):
                        duration = item['end_time'] - item['start_time']
                        st.write(f"✅ {duration:.1f} seconds")
                    else:
                        st.write(f"📊 {item['status']}")
                        
                with col3:
                    if item['status'] == 'completed':
                        if st.button("📊 Show Report", key=f"report_{item['id']}", use_container_width=True):
                            with st.expander(f"📊 {item['script_name']} Run Report", expanded=True):
                                show_run_report(item)
                    elif item['status'] in ['running', 'starting'] and item.get('pid'):
                        st.write(f"PID: {item['pid']}")
                        
                with col4:
                    if st.button("❌", key=f"remove_{item['id']}", help="remove/terminate"):
                        if item.get('pid') and item['status'] == 'running':
                            if kill_process(item['pid']):
                                st.success(f"✅ terminated process {item['pid']}")
                            else:
                                st.warning("⚠️ terminate process failed")
                        remove_from_queue(item['id'])
                        st.success(f"✅ removed from queue: {item['script_name']}")
                        time.sleep(0.5)  # short delay to ensure status update
                        st.rerun()
                
                st.markdown("---")


def load_queue():
    """load script queue"""
    queue_file = "run_queue.json"
    if os.path.exists(queue_file):
        try:
            with open(queue_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_queue(queue):
    """save script queue"""
    queue_file = "run_queue.json"
    with open(queue_file, 'w') as f:
        json.dump(queue, f, indent=2)

def add_to_queue(script_name, cmd):
    """add a running script to queue"""
    queue = load_queue()
    queue_item = {
        'id': f"{script_name}_{int(time.time())}",
        'script_name': script_name,
        'cmd': cmd,
        'status': 'starting',
        'created_time': time.time(),
        'pid': None,
        'start_time': None,
        'end_time': None
    }
    queue.append(queue_item)
    save_queue(queue)
    return queue_item

def update_queue_item(item_id, updates):
    """update queue item status"""
    queue = load_queue()
    for item in queue:
        if item['id'] == item_id:
            item.update(updates)
            break
    save_queue(queue)

def remove_from_queue(item_id):
    """remove item from queue"""
    queue = load_queue()
    queue = [item for item in queue if item['id'] != item_id]
    save_queue(queue)

def check_process_status(pid):
    """check process status"""
    try:
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            return proc.status() in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
        return False
    except:
        return False

def kill_process(pid):
    """kill process"""
    try:
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            proc.terminate()
            time.sleep(2)
            if proc.is_running():
                proc.kill()
            return True
    except:
        pass
    return False

def main():
    """main function"""
    st_config()
    create_sidebar()
    create_main_content()

if __name__ == '__main__':
    main() 