"""Home page: tutorial, system info, and pending-features banner."""
import platform, torch , sys
import streamlit as st
import psutil

from src.proj import CONST
from src.interactive.frontend.frontend import expander_subheader
from src.interactive.main.util import print_page_header

PAGE_NAME = 'home'

def estimate_text_width(text: str, font_size: int = 24) -> int:
    """Estimate the rendered pixel width of *text* at *font_size*.

    Counts Chinese characters (CJK Unified block) at 1.32× font_size and
    remaining characters at 0.66× font_size.

    Args:
        text: The string to measure.
        font_size: Font size in pixels (default 24).

    Returns:
        Estimated width in pixels.
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese_chars
    
    char_width = font_size * 0.66
    chinese_width = font_size * 1.32
    
    estimated_width = chinese_chars * chinese_width + english_chars * char_width
    return int(estimated_width)


def show_tutorial() -> None:
    """Render the tutorial expander with numbered step instructions."""
    with expander_subheader('home-tutorial' , 'Tutorial' , ':material/school:' , True ,
                            help = 'Basic Tutorial for the Project.'):
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
    
    with expander_subheader('home-system-info' , 'System Info' , ':material/computer:' , True ,
                            help = 'System Info , includes OS, memory, GPU, CPU, Python, and Streamlit version.'):
        for i , (label , value) in enumerate(options.items()):
            cols = st.columns([1,4])
            cols[0].markdown(f"***{label}***")
            cols[1].markdown(f":blue-badge[*{value}*]")
        
def show_pending_features() -> None:
    """Render warning badges for any pending features configured in preferences."""
    if not CONST.Pref.get('interactive' , 'pending_features' , []):
        return
    with expander_subheader('home-pending-features' , 'Pending Features' , ':material/pending_actions:' , True):
        for feature in CONST.Pref.get('interactive' , 'pending_features' , []):
            st.warning(feature , icon = ":material/schedule:")

def main() -> None:
    """Entry point for the home page."""
    print_page_header(PAGE_NAME)
    show_tutorial()
    show_system_info()
    show_pending_features()
    
if __name__ == '__main__':
    main() 