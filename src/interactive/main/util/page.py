import streamlit as st
from pathlib import Path
from typing import Literal

from src.proj import Proj , CONST
from src.interactive.backend import PathItem , runs_page_url

from .control import SC , set_current_page

PAGE_DIR = Path(__file__).parent.parent.joinpath('pages')
assert PAGE_DIR.exists() , f"Page directory {PAGE_DIR} does not exist"

INTRO_PAGES = ['home' , 'developer_info' , 'config_editor' , 'task_queue']

PAGE_TITLE = f":red[:material/rocket_launch:] :rainbow[{CONST.Pref.get('interactive' , 'page_title' , 'Learndl')} (_v{Proj.version}_)]"

PAGE_ICONS = {
    'home' : ':material/home:' ,
    'developer_info' : ':material/bug_report:' ,
    'config_editor' : ':material/edit_document:' ,
    'task_queue' : ':material/event_list:' ,
    'script_structure' : ':material/account_tree:' ,
}

PAGE_HELPS = {
    'home' : f"Tutorial , System Info and Links." ,
    'developer_info' : f"This is for developer only. Check boxes to select what information to show." ,
    'config_editor' : 'This File Editor is for editing selected config files. For other config files, please use the file explorer.' ,
    'task_queue' : f"Shows the entire task queue. Adjust filter to show more specific tasks." ,
    'script_structure' : f"The script structure of project runs. Click the script button to switch to script page." ,
}

SCRIPT_ICONS = {
    'check'   : ':material/question_mark:' ,
    'autorun' : ':material/schedule:' ,
    'data'    : ':material/database:' ,
    'factor'  : ':material/graph_3:' ,
    'train'   : ':material/model_training:' ,
    'test'    : ':material/experiment:' ,
    'predict' : ':material/online_prediction:' ,
    'trading' : ':material/payments:',
}

def intro_pages():
    return {page:get_intro_page(page) for page in INTRO_PAGES}

def get_intro_page(page_name : str):
    assert page_name in INTRO_PAGES , f"Page {page_name} not a valid intro page"
    if 'app_intro_pages' not in st.session_state: 
        st.session_state['app_intro_pages'] = {}
    if page_name not in st.session_state['app_intro_pages']:
        label = page_name.replace('_', ' ').title()
        icon = PAGE_ICONS[page_name]
        help = PAGE_HELPS[page_name]
        st.session_state['app_intro_pages'][page_name] = {
            'page' : st.Page(f'pages/{page_name}.py' , title = label , icon = icon) ,
            'label' : label ,
            'head' : label ,
            'icon' : icon ,
            'help' : help ,
        }
    return st.session_state['app_intro_pages'][page_name]

def script_pages():
    pages = {}
    items = [item for item in SC.path_items if item.is_file and item.level > 0]
    for item in items:
        if not runs_page_path(item.script_key).exists():
            make_script_detail_file(item)
        pages[item.script_key] = get_script_page(item.script_key)
    return pages

def get_script_page(script_key: str):
    runner = SC.get_script_runner(script_key)
    if runner.header.disabled: 
        st.error(f"Script {script_key} is disabled!")
        return {}
    if 'app_script_pages' not in st.session_state: 
        st.session_state['app_script_pages'] = {}
    
    if runner.script_key not in st.session_state['app_script_pages']:
        if runner.script_key not in SC.script_runners: 
            SC.script_runners[runner.script_key] = runner
        
        assert runs_page_path(runner.script_key).exists() , f"Script detail page {runs_page_path(runner.script_key)} does not exist"
        icon = SCRIPT_ICONS[runner.script_group]
        help = f"**Script**: *{str(runner.script)}*\n**Description**: {runner.content}"
        if runner.todo: 
            help += f"\n**TODO**: {runner.todo}"
        st.session_state['app_script_pages'][runner.script_key] = {
            'page' : st.Page(runner.page_url , title = runner.format_path , icon = icon) ,
            'group' : runner.script_group ,
            'label' : runner.format_path ,
            'head' : runner.format_path ,
            'icon' : icon ,
            'help' : help ,
            'runner' : runner ,
        }
    return st.session_state['app_script_pages'][runner.script_key]

def runs_page_path(script_key : str):
    """get runs page path"""
    return PAGE_DIR.parent.joinpath(runs_page_url(script_key))

def all_runs_page_paths():
    return [path for path in PAGE_DIR.iterdir() if path.is_file and path.name.startswith('_')]

def make_script_detail_file(item : PathItem | Path):
    """make script detail file"""
    if item.is_dir: 
        return
    if isinstance(item, Path):
        item = PathItem.from_path(item)
    with open(runs_page_path(item.script_key), 'w') as f:
        f.write(f"""
from src.interactive.main.util import show_script_detail

def main():
    show_script_detail({repr(item.script_key)}) 

if __name__ == '__main__':
    main()
""")

def remake_all_script_detail_files():
    [path.unlink() for path in all_runs_page_paths()]
    [make_script_detail_file(path) for path in PathItem.iter_folder()]

def print_page_header(page_name : str , type : Literal['intro' , 'script'] = 'intro'):
    set_current_page(page_name)
    SC.switch_page(page_name)
    if type == 'intro':
        script_key = None
        self_page = get_intro_page(page_name) 
    elif type == 'script':
        script_key = page_name
        self_page = get_script_page(script_key)
        if self_page is None:
            st.error(f"Script {script_key} not not enabled")
            return
    else:
        raise ValueError(f"type {type} should be 'intro' or 'script'")
    
    # st.title(PAGE_TITLE)
    st.header(f"*_:red[{self_page['icon']}] :rainbow[{self_page['head']}]_*" , help = self_page['help'])
    # if 'control-panel' not in st.session_state:
    #     st.session_state['control-panel'] = ControlPanel()
    # st.session_state['control-panel'].show(script_key = script_key)
    SC.get_control_panel().show(script_key = script_key)
    