import streamlit as st
from pathlib import Path
import re
from src_app.backend import ScriptRunner , PathItem
from .control import SC
from .basic import __version__ , __page_title__

PAGE_DIR = Path(__file__).parent.parent.joinpath('pages')
assert PAGE_DIR.exists() , f"Page directory {PAGE_DIR} does not exist"
    
def menu_pages():
    if not SC.menu_pages: 
        SC.menu_pages = {
            'home': home_page() ,
            'dev_info': developer_info_page() ,
            'config_editor': config_editor_page() ,
            'task_queue': task_queue_page() ,
            'scripts': script_structure_page() ,
        }
    return SC.menu_pages

def script_pages():
    if not SC.script_pages: 
        pages = {}
        items = SC.path_items
        for item in items:
            if not item.is_dir and item.level > 0:
                runner = item.script_runner()
                make_script_detail_file(item)
                pages[runner.script_key] = get_script_page(runner)
        SC.script_pages = pages
    return SC.script_pages

def home_page():
    return {
        'page' : st.Page('pages/home.py' , title = 'Home' , icon = ':material/home:') ,
        'label' : 'Home' ,
        'icon' : ':material/home:' ,
        'help' : f":material/info: **{__page_title__}** \n*{__version__}*" ,
    }
def developer_info_page():
    return {
        'page' : st.Page('pages/developer_info.py' , title = 'Developer Info' , icon = ':material/bug_report:') ,
        'label' : 'Developer Info' ,
        'icon' : ':material/bug_report:' ,
        'help' : f":material/info: **Developer Info**" ,
    }
def config_editor_page():
    return {
        'page' : st.Page('pages/config_editor.py' , title = 'Config Editor' , icon = ':material/edit_document:') ,
        'label' : 'Config Editor' ,
        'icon' : ':material/edit_document:' ,
        'help' : f":material/info: **Config Editor**" ,
    }
def task_queue_page():
    return {
        'page' : st.Page('pages/task_queue.py' , title = 'Task Queue' , icon = ':material/event_list:') ,
        'label' : 'Task Queue' ,
        'icon' : ':material/event_list:' ,
        'help' : f":material/info: **Task Queue**" ,
    }
def script_structure_page():
    return {
        'page' : st.Page('pages/script_structure.py' , title = 'Script Structure' , icon = ':material/folder_open:') ,
        'label' : 'Script Structure' ,
        'icon' : ':material/folder_open:' ,
        'help' : f":material/info: **Script Structure**" ,
    }
def get_script_page(runner: ScriptRunner):
    if runner.header.disabled: 
        print(f'{runner.script_key} is disabled')
        return
    if runner.script_key not in SC.script_runners:
        SC.script_runners[runner.script_key] = runner
    assert SC.script_runners , "script runners are not initialized"
    
    assert runs_page_path(runner.script_key).exists() , f"Script detail page {runs_page_path(runner.script_key)} does not exist"
    icon = {
        'check' : ':material/question_mark:' ,
        'autorun' : ':material/schedule:' ,
        'research' : ':material/experiment:' ,
        'trading' : ':material/payments:',
    }[runner.script_group]
    return {
        'page' : st.Page(runs_page_url(runner.script_key) , title = runner.format_path , icon = icon) ,
        'group' : runner.script_group ,
        'label' : runner.format_path ,
        'icon' : icon ,
        'help' : f":material/info: **{runner.content}** \n*{str(runner.script)}*" ,
        }

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