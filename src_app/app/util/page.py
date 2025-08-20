import streamlit as st
from pathlib import Path
from typing import Literal
import re
from src_app.backend import PathItem
from .control import SC
from .basic import VERSION , PAGE_TITLE

PAGE_DIR = Path(__file__).parent.parent.joinpath('pages')
assert PAGE_DIR.exists() , f"Page directory {PAGE_DIR} does not exist"

INTRO_PAGES = ['home' , 'developer_info' , 'config_editor' , 'task_queue']

PAGE_TITLES = {
    'home' : f":rainbow[:material/rocket_launch: {PAGE_TITLE} (_v{VERSION}_)]"
}

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
    'check' : ':material/question_mark:' ,
    'autorun' : ':material/schedule:' ,
    'research' : ':material/experiment:' ,
    'trading' : ':material/payments:',
}

def intro_pages():
    return {page:get_intro_page(page) for page in INTRO_PAGES}

def get_intro_page(page_name : str):
    assert page_name in INTRO_PAGES , f"Page {page_name} not a valid intro page"
    if 'app_intro_pages' not in st.session_state: st.session_state['app_intro_pages'] = {}
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
    items = SC.path_items
    for item in items:
        if not item.is_dir and item.level > 0:
            make_script_detail_file(item)
            pages[item.script_key] = get_script_page(item.script_key)
    return pages

def get_script_page(script_key: str):
    runner = SC.get_script_runner(script_key)
    if runner.header.disabled: 
        st.error(f"Script {script_key} is disabled!")
        return {}
    if 'app_script_pages' not in st.session_state: st.session_state['app_script_pages'] = {}
    
    if runner.script_key not in st.session_state['app_script_pages']:
        if runner.script_key not in SC.script_runners: SC.script_runners[runner.script_key] = runner
        
        assert runs_page_path(runner.script_key).exists() , f"Script detail page {runs_page_path(runner.script_key)} does not exist"
        icon = SCRIPT_ICONS[runner.script_group]
        help = f"**Script**: *{str(runner.script)}*\n**Description**: {runner.content}"
        if runner.todo: help += f"\n**TODO**: {runner.todo}"
        st.session_state['app_script_pages'][runner.script_key] = {
            'page' : st.Page(runs_page_url(runner.script_key) , title = runner.format_path , icon = icon) ,
            'group' : runner.script_group ,
            'label' : runner.format_path ,
            'head' : runner.format_path ,
            'icon' : icon ,
            'help' : help ,
        }
    return st.session_state['app_script_pages'][runner.script_key]

def runs_page_url(script_key : str):
    """get runs page url"""
    return "pages/_" + re.sub(r'[/\\]', '_', script_key)

def runs_page_path(script_key : str):
    """get runs page path"""
    return PAGE_DIR.joinpath(runs_page_url(script_key).split('/')[-1])

def make_script_detail_file(item : PathItem):
    """make script detail file"""
    if item.is_dir: return
    with open(runs_page_path(item.script_key), 'w') as f:
        f.write(f"""
from util import show_script_detail , set_current_page

def main():
    set_current_page({repr(item.script_key)}) 
    show_script_detail({repr(item.script_key)}) 

if __name__ == '__main__':
    main()
""")
        
def print_page_header(page_name : str , type : Literal['intro' , 'script'] = 'intro'):
    if type == 'intro':
        self_page = get_intro_page(page_name) 
    elif type == 'script':
        script_key = page_name
        self_page = get_script_page(script_key)
        if self_page is None:
            st.error(f"Script {script_key} not not enabled")
            return
    else:
        raise ValueError(f"type {type} should be 'intro' or 'script'")
    title = PAGE_TITLES.get(page_name , f":rainbow[{self_page['icon']} {self_page['head']}]")
    helps = self_page['help'].split('\n')
    st.session_state['box-title'].title(title)
    # st.session_state['box-main-button'].write('')
    for h in helps: st.warning(h , icon = ":material/info:")