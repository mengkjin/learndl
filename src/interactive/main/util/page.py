import streamlit as st
from pathlib import Path
from typing import Literal
import re , subprocess
from abc import abstractmethod , ABC

from src.proj import Proj , MACHINE 
from src.proj.util import Options
from src.interactive.backend import PathItem
from src.interactive.backend import ScriptRunner

from .control import SC , set_current_page

PAGE_DIR = Path(__file__).parent.parent.joinpath('pages')
assert PAGE_DIR.exists() , f"Page directory {PAGE_DIR} does not exist"

INTRO_PAGES = ['home' , 'developer_info' , 'config_editor' , 'task_queue']

PAGE_TITLE = f":rainbow[:material/rocket_launch: {Proj.Conf.Interactive.page_title} (_v{Proj.Conf.Interactive.version}_)]"

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
            'page' : st.Page(runs_page_url(runner.script_key) , title = runner.format_path , icon = icon) ,
            'group' : runner.script_group ,
            'label' : runner.format_path ,
            'head' : runner.format_path ,
            'icon' : icon ,
            'help' : help ,
            'runner' : runner ,
        }
    return st.session_state['app_script_pages'][runner.script_key]

def runs_page_url(script_key : str):
    """get runs page url"""
    return "pages/_" + re.sub(r'[/\\]', '_', script_key)

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
from util import show_script_detail

def main():
    show_script_detail({repr(item.script_key)}) 

if __name__ == '__main__':
    main()
""")

def remake_all_script_detail_files():
    [path.unlink() for path in all_runs_page_paths()]
    [make_script_detail_file(path) for path in PathItem.iter_folder()]

class ControlPanelButton(ABC):
    """control panel button"""
    key : str = ''
    icon : str = ''
    title : str = ''

    @abstractmethod
    def button(self , script_key : str | None = None):
        ...

    def refresh(self , *args , **kwargs):
        pass

    def show(self , script_key : str | None = None):
        if self.key not in st.session_state:
            st.session_state[self.key] = st.empty()
        with st.session_state[self.key]:
            with st.container():
                self.button(script_key = script_key)
                self.print_title()

    def print_title(self):
        body = f"""
        <div style="
            margin-bottom: 0px;
            margin-top: -10px;
            padding: 0 0 20px 0;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        ">{self.title.upper()}</div>
        """       
        st.markdown(body , unsafe_allow_html = True)

class ScriptRunnerRunButton(ControlPanelButton):
    key = f"script-runner-run"
    icon = f":material/mode_off_on:"
    title = f"Run Script"

    def button(self , script_key : str | None = None):
        help = f"Please Choose a Script to Run First" if script_key is None else f"Please Fill Required Parameters"
        st.button(self.icon, key=f'{self.key}-disabled' , help = help)

    def refresh(self , runner : ScriptRunner):
        with st.session_state[self.key]:
            if SC.param_inputs_form is None:
                raise ValueError("ParamInputsForm is not initialized")
            params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
            
            if SC.get_script_runner_validity(params):
                disabled = False
                preview_cmd = SC.get_script_runner_cmd(runner , params)
                if preview_cmd: 
                    help_text = preview_cmd
                else:
                    help_text = f"Parameters valid, run {runner.script_key}"
                button_key = f"{self.key}-enabled-{runner.script_key}"
            else:
                disabled = True
                help_text = f"Parameters invalid, please check required ones"
                button_key = f"{self.key}-disabled-{runner.script_key}"

            with st.container():
                st.button(self.icon, key=button_key , 
                        help = help_text , disabled = disabled , 
                        on_click = SC.click_script_runner_run , args = (runner, params)) 
                self.print_title()

class GlobalScriptLatestTaskButton(ControlPanelButton):
    key = f"global-script-latest-task"
    icon = f":material/reply_all:"
    title = f"Latest for All"

    def button(self , script_key : str | None = None):
        item = SC.get_latest_task_item()
        if item is None:
            st.button(self.icon, key=f"{self.key}-disabled" , 
                    help = "Please Run a Task First" , disabled = True)
        else:
            if st.button(self.icon, key=f"{self.key}-enabled-{item.id}" , 
                        help = f":blue[**Show Latest Task**]: {item.id}" , 
                        on_click = SC.click_show_complete_report , args = (item,) ,
                        disabled = False):
                if SC.current_page_name != repr(item.script_key):
                    st.switch_page(runs_page_url(item.script_key))
                else:
                    #from .script_detail import show_report_main
                    #show_report_main(SC.get_script_runner(item.script_key))
                    st.rerun()

class CurrentScriptLatestTaskButton(ControlPanelButton):
    key = f"current-script-latest-task"
    icon = f":material/reply:"
    title = f"Current Latest"

    def button(self , script_key : str | None = None):
        item = SC.get_latest_task_item(script_key) if script_key is not None else None
        if item is None:
            st.button(self.icon, key=f"{self.key}-disabled" , 
                        help = "Please Run a Task of This Script First" if script_key is not None else "Please Choose a Script First" , disabled = True)
        else:
            if st.button(self.icon, key=f"{self.key}-enabled-{item.id}" , 
                        help = f":blue[**Show Latest Task of This Script**]: {item.id}" , 
                        on_click = SC.click_show_complete_report , args = (item,) ,
                        disabled = False):
                #from .script_detail import show_report_main
                #show_report_main(SC.get_script_runner(item.script_key))
                st.rerun()

class ControlRefreshInteractiveButton(ControlPanelButton):
    key = f"control-refresh-interactive"
    icon = f":material/refresh:"
    title = f"Refresh All"

    def button(self , script_key : str | None = None):
        st.button(self.icon, key=f"{self.key}-enabled" , help = "Refresh Task Queue / Options / Scripts" , 
                  on_click = self.refresh_all , disabled = False)
    
    def refresh_all(self):
        with st.spinner("Refreshing..."):
            with Proj.Silence:
                Options.update()
                remake_all_script_detail_files()
        SC.rerun()
        st.rerun()

class ControlGitClearPullButton(ControlPanelButton):
    key = f"control-git-clear-pull"
    icon = f":material/cloud_download:"
    title = f"Git Pull"

    def button(self , script_key : str | None = None):
        if MACHINE.platform_coding:
            st.button(self.icon, key=f"{self.key}-disabled" , help = f"Git Pull is not available on coding platform {MACHINE.name}" , disabled = True)
        else:
            st.button(self.icon, key=f"{self.key}-enabled" , help = "Clear Local Changes and Pull Latest Code" , disabled = False, on_click = self.clear_git_pull)
        
    def clear_git_pull(self):
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        else:
            import shutil
            from src.proj import PATH , Logger

            subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
            subprocess.run(['git', 'clean', '-fd'], check=True)
            subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True)
            
            for folder in [*PATH.main.joinpath('src').rglob('*/') , *PATH.main.joinpath('configs').rglob('*/')][::-1]:
                if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
                    subfiles = [x for x in folder.rglob('*') if x.is_file()]
                    if not len(subfiles):
                        Logger.stdout(f"Removing empty folder: {folder}")
                        folder.rmdir()
                    else:
                        if all([x.suffix == '.pyc' for x in subfiles]):
                            Logger.stdout(f"Removing folder with only pyc files: {folder}")
                            shutil.rmtree(folder)
                        else:
                            Logger.error(f"Error removing folder: {folder}:")
                            Logger.error(f"Subfiles: {subfiles}")
            Logger.success("Git Pull Finished")
class ControlPanel:
    """control panel"""
    control_panel_key = "page-control-panel"
    buttons : dict[str, ControlPanelButton] = {
        'script-runner-run' : ScriptRunnerRunButton(),
        'global-script-latest-task' : GlobalScriptLatestTaskButton(),
        'current-script-latest-task' : CurrentScriptLatestTaskButton(),
        'control-refresh-interactive' : ControlRefreshInteractiveButton(),
        'control-git-clear-pull' : ControlGitClearPullButton(),
    }
    
    def show(self , script_key : str | None = None):
        with st.container(key = self.control_panel_key):
            buttons , settings = st.tabs(['**Global Control**' , '**Global Settings**'])
            with buttons:
                self.show_buttons(script_key = script_key)
            with settings:
                self.show_settings(script_key = script_key)

    def show_buttons(self , script_key : str | None = None):
        with st.container(key = f"{self.control_panel_key}-buttons"):
            cols = st.columns(len(self.buttons) , gap = 'small' , vertical_alignment = 'center')
            for col , button in zip(cols, self.buttons.values()):
                with col:
                    button.show(script_key = script_key)

    def show_settings(self , script_key : str | None = None):
        with st.container(key = f"{self.control_panel_key}-settings"):
            verbosity , email , mode = st.columns(3 , gap = 'small' , vertical_alignment = 'center')
            with verbosity:
                # max verbosity, yes for 10 , no for 0 , None for default (2 if not set), passed to script params
                st.segmented_control("**:blue[Max Verbosity]**" , ['yes' , 'no'] , default = None , 
                                     key = 'global-settings-maxvb'  , 
                                     help="""Should use max verbosity or min? Not selected will use default.""")

            with email:
                # email notification, yes , no , None for default, passed to script params
                st.segmented_control("**:blue[Send Email]**" , ['yes' , 'no'] , default = None , 
                                     key = 'global-settings-email'  , 
                                     help="""If email after the script is complete? Not selected will use script header value.""")

            with mode:    
                # run mode, shell , os , or default , used in SessionControl.click_script_runner_run()
                st.segmented_control("**:blue[Run Mode]**" , ['shell' , 'os'] , default = None , 
                                     key = 'global-settings-mode'  , 
                                     help="""Which mode should the script be running in?
                                     :blue[**shell**] will start a commend terminal to run;
                                     :blue[**os**] will run in backend.
                                     Not selected will use script header value.""")
          
          
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
    st.header(f"*_:rainbow[{self_page['icon']} {self_page['head']}]_*" , help = self_page['help'])
    if 'control-panel' not in st.session_state:
        st.session_state['control-panel'] = ControlPanel()
    st.session_state['control-panel'].show(script_key = script_key)
    