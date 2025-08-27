from util import PENDING_FEATURES , set_current_page , show_sidebar_buttons , intro_pages , print_page_header
import platform, torch , sys , re
import streamlit as st
import psutil

from src_app.frontend.frontend import expander_subheader
from src_app.backend import ScriptRunner
from util import SC , set_current_page , show_sidebar_buttons , get_script_page , print_page_header

PAGE_NAME = 'home'

def estimate_text_width(text, font_size=24):
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese_chars
    
    char_width = font_size * 0.66
    chinese_width = font_size * 1.32
    
    estimated_width = chinese_chars * chinese_width + english_chars * char_width
    return int(estimated_width)

def expander_subheader(key : str , label : str , icon : str | None = None , expanded = False , 
                       height : int | None = None , help : str | None = None , status = False , color = 'blue'):
    
    container_key = f'{key.replace(" " , "-").lower()}-special-expander-' + ('status' if status else 'expander')
    with st.container():
        if help is not None:
            help_icon = """<span role="img" aria-label="mode_off_on icon" translate="no" style="display: inline-block; 
                            font-family: &quot;Material Symbols Rounded&quot;; 
                            user-select: none; 
                            vertical-align: bottom; 
                            overflow-wrap: normal;">help</span>"""
            margin_left = 10
            if icon is not None: margin_left += 34
            if status:           margin_left += 28
            margin_left += estimate_text_width(label.upper())
            st.markdown(f"""
            <div class="expander-help-container">
                <div class="help-tooltip">
                    {help}
                </div>
                <span class="help-icon" style="margin-left: {margin_left}px;">
                    {help_icon}
                </span>
            </div>
            """, unsafe_allow_html=True)
        container = st.container(key = container_key)
        full_label = label if icon is None else f'{icon} {label}'
        if status:
            exp_container = container.status(f" :{color}[{full_label}]" , expanded = expanded)
        else:
            exp_container = container.expander(f":{color}[{full_label}]" , expanded = expanded).container(height = height)
        
    return exp_container

def show_tutorial():
    with expander_subheader('home-tutorial' , 'Tutorial' , ':material/school:' , True ,
                            help = 'Basic Tutorial for the Project.'):
        st.markdown("""
        1. :blue[:material/settings:] Click the script button to expand the parameter settings
        2. :green[:material/mode_off_on:] Fill in the necessary parameters and click Run
        3. :rainbow[:material/bar_chart:] View the running report and generated files
        4. :gray[:material/file_present:] Preview the generated HTML/PDF files
        """)

def show_system_info():
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
    
    with expander_subheader('home-system-info' , 'System Info' , ':material/computer:' , True ,
                            help = 'System Info , includes OS, memory, GPU, CPU, Python, and Streamlit version.'):
        cols = st.columns(len(options))
        for i , (label , value) in enumerate(options.items()):
            cols[i].metric(f"{label}" , value)
        
def show_pending_features():
    if not PENDING_FEATURES: return
    with expander_subheader('home-pending-features' , 'Pending Features' , ':material/pending_actions:' , True):
        for feature in PENDING_FEATURES:
            st.warning(feature , icon = ":material/schedule:")

def show_intro_pages():
    with expander_subheader('home-intro-pages' , 'Other Intro Pages' , ':material/outdoor_garden:' , True , 
                            help = 'Click to Switch to Other Intro Pages.'):
        pages = {k:v for k,v in intro_pages().items() if k != PAGE_NAME}
        cols = st.columns(len(pages))
        for col , (name , page) in zip(cols , pages.items()):
            button = col.button(page['label'] , icon = page['icon'] , key = f"intro-page-{name}")
            if button: st.switch_page(page['page'])


def show_script_structure():
    """show folder content recursively"""  
    with expander_subheader('home-script-structure' , 'Script Structure' , ':material/account_tree:' , True , 
                            help = 'Script Structure of the Project, Click to Switch to Detailed Script Page.'):
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

def show_script_runner(runner: ScriptRunner):
    """show single script runner"""
    if runner.script_key not in SC.script_runners: SC.script_runners[runner.script_key] = runner
    
    page = get_script_page(runner.script_key)
    if page is None: return
    
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


def main():
    set_current_page(PAGE_NAME)
    print_page_header(PAGE_NAME)
    show_tutorial()
    show_system_info()
    show_pending_features()
    show_intro_pages()
    show_script_structure()
    show_sidebar_buttons()
    
if __name__ == '__main__':
    main() 