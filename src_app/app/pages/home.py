from util import PENDING_FEATURES , set_current_page , show_run_button_sidebar , intro_pages , print_page_header
import platform, torch , sys
import streamlit as st
import psutil

PAGE_NAME = 'home'

def show_other_intro_pages():
    with st.container(key = "home-intro-pages"):
        st.header(":material/outdoor_garden: Other Intro Pages" , divider = 'grey')
        pages = {k:v for k,v in intro_pages().items() if k != PAGE_NAME}
        cols = st.columns(len(pages))
        for col , (name , page) in zip(cols , pages.items()):
            button = col.button(page['label'] , icon = page['icon'] , key = f"intro-page-{name}")
            if button: st.switch_page(page['page'])

def show_tutorial():
    st.subheader(":blue[:material/school: Tutorial]" , divider = 'grey')
    st.markdown("""
    1. :blue[:material/settings:] Click the script button to expand the parameter settings
    2. :green[:material/mode_off_on:] Fill in the necessary parameters and click Run
    3. :rainbow[:material/bar_chart:] View the running report and generated files
    4. :gray[:material/file_present:] Preview the generated HTML/PDF files
    """)

def show_system_info():
    st.subheader(":blue[:material/computer: System Info]" , divider = 'grey')
    options : dict[str, str] = {}
    # os
    options[':material/keyboard_command_key: **OS**'] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    # memory
    mem = psutil.virtual_memory()
    options[':material/memory: **Memory Usage**'] = \
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
            gpu_info = "**GPU Usage (MPS)**" , f"{used:.1f} / {recommend:.1f} GB ({used / recommend * 100:.1f}%)"
        else:
            gpu_info = "**GPU Usage (MPS)**" , f"{used:.1f} GB Used"
    else:
        gpu_info = "**GPU Usage (None)**" , "No GPU"
    options[f':material/memory_alt: **{gpu_info[0]}**'] = f"{gpu_info[1]}"
    # cpu
    options[':material/select_all: **CPU Usage**'] = f"{psutil.cpu_percent():.1f}%"
    # python
    options[':material/commit: **Python Version**'] = f"{sys.version.split(' ')[0]}"
    # streamlit
    options[':material/commit: **Streamlit Version**'] = f"{st.__version__}"
    
    cols = st.columns(len(options))
    for i , (label , value) in enumerate(options.items()):
        cols[i].metric(f"{label}" , value)
        
def show_pending_features(pending_features : list[str] | None = None):
    st.subheader(":blue[:material/pending_actions: Pending Features]" , divider = 'grey')
    if pending_features is None:
        st.warning("No pending features" , icon = ":material/schedule:")
        return
    for feature in pending_features:
        st.warning(feature , icon = ":material/schedule:")

def main():
    set_current_page(PAGE_NAME)
    print_page_header(PAGE_NAME)
    # show_other_intro_pages()
    show_tutorial()
    show_system_info()
    show_pending_features(PENDING_FEATURES)
    show_run_button_sidebar()
    
if __name__ == '__main__':
    main() 