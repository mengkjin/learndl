import platform, torch , sys
import streamlit as st
import psutil

PENDING_FEATURES = []

def basic_info():
    show_tutorial()
    show_system_info()
    show_pending_features()

def show_tutorial():
    st.header(":material/school: Tutorial" , divider = 'grey')
    st.markdown("""
    1. :blue[:material/settings:] Click the script button to expand the parameter settings
    2. :green[:material/mode_off_on:] Fill in the necessary parameters and click Run
    3. :rainbow[:material/bar_chart:] View the running report and generated files
    4. :gray[:material/file_present:] Preview the generated HTML/PDF files
    """)

def show_system_info():
    st.header(":material/computer: System Info" , divider = 'grey')
    options : dict[str, str] = {}
    # os
    options[':material/keyboard_command_key: **OS**'] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    # memory
    options[':material/memory: **Memory Usage**'] = \
        f"{psutil.virtual_memory().used / 1024**3:.1f} GB / {psutil.virtual_memory().total / 1024**3:.1f} GB ({psutil.virtual_memory().percent:.1f}%)"
    # gpu
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = "**GPU Usage (CUDA)**" , f"{used:.1f} / {total:.1f} GB ({used / total * 100:.1f}%)"
    elif torch.backends.mps.is_available():
        used = torch.mps.current_allocated_memory() / 1024**3
        if torch.__version__ >= '2.3.0':
            recommend = torch.mps.recommended_max_memory() / 1024**3 # type:ignore
            gpu_info = "**GPU Usage (MPS)**" , f"{used:.1f} / {recommend:.1f} GB ({used / recommend * 100:.1f}%)"
        else:
            gpu_info = "**GPU Usage (MPS)**" , f"{used:.1f} GB Used"
    else:
        gpu_info = "**GPU Usage (None)**" , "No GPU"
    options[f':material/memory_alt: **{gpu_info[0]}**'] = f"{gpu_info[1]}"
    # cpu
    options[':material/select_all: **CPU Usage**'] = f"{psutil.cpu_percent():.1f}%"
    # python
    options[':material/commit: **Python Version**'] = f"{sys.version}"
    # streamlit
    options[':material/commit: **Streamlit Version**'] = f"{st.__version__}"
    
    for label , value in options.items():
        st.metric(f":blue[{label}]" , value)
        
def show_pending_features():
    st.header(":material/pending_actions: Pending Features" , divider = 'grey')
    for feature in PENDING_FEATURES:
        st.warning(feature , icon = ":material/schedule:")

    