import streamlit as st
import os , subprocess

from typing import Any , Literal, Callable
from pathlib import Path

from src.interactive.backend import (
    ScriptRunner , TaskItem
)

from src.interactive.frontend import (
    FilePreviewer , YAMLFileEditor , ColoredText , expander_subheader , ParamInputsForm
)

from src.proj import PATH , MACHINE

from util.control import SC , set_current_page
from util.page import get_script_page , print_page_header , runs_page_url

def on_first_page(max_page : int):
    if st.session_state.get('choose-task-page') == 1: 
        return
    st.session_state['choose-task-page'] = 1
def on_last_page(max_page : int):
    if st.session_state.get('choose-task-page') == max_page: 
        return
    st.session_state['choose-task-page'] = max_page
def on_prev_page(max_page : int):
    if st.session_state.get('choose-task-page') == 1: 
        return
    st.session_state['choose-task-page'] = max((st.session_state.get('choose-task-page') or 1) - 1, 1)
def on_next_page(max_page : int):
    if st.session_state.get('choose-task-page') == max_page: 
        return
    st.session_state['choose-task-page'] = (st.session_state.get('choose-task-page') or 1) + 1

def show_script_detail(script_key : str):
    """show main part"""
    set_current_page(script_key)
    runner = SC.get_script_runner(script_key)
    page = get_script_page(script_key)
    if page is None: 
        return
    
    print_page_header(script_key , 'script')  
    show_script_task_selector(runner)
    show_param_settings(runner)
    show_main_buttons(runner)
    show_sidebar_buttons(runner)
    show_report_main(runner)
    

def clear_and_show(show_func : Callable):
    name = show_func.__name__.removeprefix('show_').replace('_', '-')
    placeholder_key = f'{name}-placeholder'
    def wrapper(*args , **kwargs):
        if placeholder_key not in st.session_state: 
            st.session_state[placeholder_key] = st.empty()
        with st.session_state[placeholder_key]:
            show_func(*args , **kwargs)
    return wrapper

@clear_and_show
def show_script_task_selector(runner : ScriptRunner):
    """show script task selector"""
    script_queue = SC.task_queue.filter(file = [runner.path.path])
    is_empty = not script_queue
    wkey = 'script-task-selector'
    header = 'Historical Tasks'
    icon = ':material/history:'
    if is_empty: 
        help = "Empty. No Past Task Item is Recorded for This Script."
    else:
        help = "Choose Any Task Item from the Expanders Below."
    subheader = expander_subheader(wkey , header , icon , False , help = help)

    with subheader:
        if is_empty: 
            return
        
        status_options = ["All" , "Running" , "Complete" , "Error"]
        source_options = ["All" , "Py" , "App" ,"Bash" , "Other"]
        status = st.radio(":blue-badge[**Running Status**]" , status_options , key = "task-filter-status", horizontal = True)
        source = st.radio(":blue-badge[**Script Source**]" , source_options , key = "task-filter-source", horizontal = True)
        queue = SC.task_queue.filter(status = status , source = source , queue = script_queue)

        cols = st.columns(2)
        with cols[0].container(key = f"task-stats-unfiltered-container"):
            st.info(f"**:material/bar_chart: Stats: All Tasks of This Script**")
            st.caption(f":blue-badge[:material/update: Status] {SC.task_queue.status_message(script_queue)}")
            st.caption(f":blue-badge[:material/distance: Source] {SC.task_queue.source_message(script_queue)}")
            
        with cols[1].container(key = f"task-stats-filtered-stats-container"):
            st.info(f"**:material/bar_chart: Stats: Filtered Tasks**")
            st.caption(f":blue-badge[:material/update: Status] {SC.task_queue.status_message(queue)}")
            st.caption(f":blue-badge[:material/distance: Source] {SC.task_queue.source_message(queue)}")
        
        item_ids = list(queue.keys())
        if SC.current_task_item is not None and SC.current_task_item in item_ids:
            choose_index = item_ids.index(SC.current_task_item)
        else:
            choose_index = None
        
        page_option_cols = st.columns(2)

        with page_option_cols[0].container(key = f"choose-task-num-per-page-container"):
            cols = st.columns(2 , vertical_alignment = "center")
            cols[0].info('**Tasks per Page**')
            num_per_page = cols[1].selectbox('select num per page', [5 , 20 , 50 , 100 , 500], format_func = lambda x: f'{x} Tasks/Page', 
                                index = 1 , key = f"choose-task-item-num-per-page" , label_visibility = 'collapsed')
            
        with page_option_cols[1].container(key = f"choose-task-page-container"):
            max_page = (len(item_ids) - 1) // num_per_page + 1
            page_options = list(range(1, max_page + 1))
            index_page = choose_index // num_per_page if choose_index is not None else 0
            page_cols = st.columns([7, 1, 1, 3, 1, 1] , gap = None , vertical_alignment = "center")
            page_cols[0].info(f'**Select Page (1 ~ {max_page})**')
            page_cols[1].button(":material/first_page:", key = f"choose-task-page-first", on_click = on_first_page , args = (max_page,))
            page_cols[2].button(":material/chevron_left:", key = f"choose-task-page-prev", on_click = on_prev_page , args = (max_page,))
            page_cols[3].selectbox('select page', page_options, key = f"choose-task-page" , index = index_page , 
                                   placeholder = f'Page', label_visibility = 'collapsed')
            page_cols[4].button(":material/chevron_right:", key = f"choose-task-page-next", on_click = on_next_page , args = (max_page,))
            page_cols[5].button(":material/last_page:", key = f"choose-task-page-last", on_click = on_last_page , args = (max_page,))
        
        with st.container(key = f"choose-task-item-container"):
            current_page = st.session_state.get('choose-task-page') or 1
            options = item_ids[(current_page - 1) * num_per_page : current_page * num_per_page]
            default_index = (choose_index % num_per_page) if choose_index is not None else None
            show_queue_item_list(runner , script_queue , options , default_index , type = 'buttons')
        
        if SC.current_task_item:
            st.success(f"Task Item {SC.current_task_item} chosen" , icon = ":material/check_circle:")

def show_queue_item_list(runner : ScriptRunner , queue : dict[str, TaskItem] , options : list[str] , default_index : int | None = None ,
                         type : Literal['multiselect' , 'buttons'] = 'buttons'):
    """show queue item list"""
    if type == 'multiselect':
        format_dict = {item.id : item.button_str_long(i + 1, plain_text = True).strip() 
                       for i, item in enumerate(queue.values())}
        st.selectbox("Choose Task Item from Queue", 
                    options = options, 
                    index = default_index,
                    format_func = lambda x: format_dict[x],
                    key = f"choose-item-selectbox" , 
                    help = "Choose a Task Item from Filtered Queue" ,
                    placeholder = "Choose a Task Item from Filtered Queue",
                    on_change = SC.click_choose_item_selectbox , 
                    label_visibility = 'collapsed')
    elif type == 'buttons':
        indexes = {item_id : i for i, item_id in enumerate(queue.keys())}
        for item_id in options:
            item = queue[item_id]
            index = indexes[item_id]
            placeholder = st.empty()
            container = placeholder.container(key = f"script-queue-item-container-{item_id}")
            with container:
                cols = st.columns([18, .5,.5,.5,.5] , gap = None, vertical_alignment = "center")
                key = f"script-click-content-{item_id}"
                if item_id == SC.current_task_item: 
                    key += "-selected"
                with cols[0]:
                    if st.button(item.button_str_long(index + 1),  
                                help=item.button_help_text() , key=key , 
                                use_container_width=True , on_click = SC.click_choose_item_selectbox , args = (item_id,)):
                        show_script_task_selector(runner)

                cols[3].button(":violet-badge[:material/remove:]", 
                            key=f"script-queue-item-delist-{item_id}", help="Delist from Queue", 
                            on_click = SC.click_queue_delist_item , args = (queue[item_id],))
                cols[4].button(":red-badge[:material/cancel:]", 
                            key=f"script-queue-item-remove-{item_id}", help="Delete from Database (Irreversible)", 
                            on_click = SC.click_queue_remove_item , args = (queue[item_id],))
    else:
        raise ValueError(f"Invalid type: {type}")

@clear_and_show
def show_param_settings(runner : ScriptRunner):
    if runner.disabled:
        st.error(f":material/disabled_by_default: This script is disabled")
        return
    param_inputs = runner.header.get_param_inputs()
    
    is_empty = not param_inputs
    wkey = 'script-param-setting'
    header = 'Parameters Setting'
    icon = ':material/settings:'
    if is_empty:
        help = "Empty. No Parameter is Required for This Script."
    else:
        help = "Input Parameters for This Script in the Expanders Below, Mind the Required Ones."
    subheader = expander_subheader(wkey , header , icon , True , help = help)

    with subheader:
        param_controls = st.empty()
        SC.param_inputs_form = ParamInputsForm(runner , SC.script_params_cache , SC.get_task_item(SC.current_task_item)).init_param_inputs()
        if is_empty: 
            return

        cols = param_controls.columns(4)
        with cols[0]:
            if st.button(":blue-badge[:material/refresh: **Reset Parameters**]", key = f"param-inputs-form-reset-param-button" , help = "Reset Parameters to Default" , type = 'tertiary'):
                SC.script_params_cache.clear_script_cache(runner.script_key)
                SC.current_task_item = None
                st.rerun()

        with cols[1]:
            if st.button(":blue-badge[:material/history: **Last Parameters**]", key = f"param-inputs-form-last-param-button" , help = "Set Parameters to Latest Task's Parameters" , type = 'tertiary'):
                item = SC.get_latest_task_item(runner.script_key)
                if isinstance(SC.param_inputs_form, ParamInputsForm) and item is not None:
                    item_params = SC.param_inputs_form.cmd_to_param_values(cmd = item.cmd)
                    SC.script_params_cache.update_script_cache(runner.script_key, 'value', item_params)
                    st.rerun()

        params = SC.param_inputs_form.param_values
        if runner.header.file_editor:
            with st.expander(runner.header.file_editor.get('name', 'File Editor') , expanded = False , icon = ":material/edit_document:"):
                path = conditional_path(runner.header.file_editor['path'], params)
                file_editor = YAMLFileEditor('param-settings-file-editor', 
                                            file_root=path , file_input=False , 
                                            height = runner.header.file_editor.get('height'))
                file_editor.show_yaml_editor()
        if runner.header.file_previewer:
            with st.expander(runner.header.file_previewer.get('name', 'File Previewer') , expanded = False , icon = ":material/file_present:"):
                path = conditional_path(runner.header.file_previewer['path'], params)
                file_previewer = FilePreviewer(path , height = runner.header.file_previewer.get('height'))
                file_previewer.preview()

def conditional_path(format_str : str , params : dict[str, Any] , root = PATH.main):
    params = params | {'PATH' : PATH , 'MACHINE' : MACHINE}
    if '|' in format_str:
        format_strs = [s.strip() for s in format_str.split('|')]
        for s in format_strs:
            path = s.format(**params)
            if Path(path).exists() or root.joinpath(path).exists():
                return path
        return path
    else:
        return format_str.strip().format(**params)

def run_button_button(runner : ScriptRunner | None , sidebar = False):
    if runner is None:
        disabled = True
        help_text = f"Please Choose a Script to Run First"
        button_key = f"script-runner-run-disabled-not-selected"
        params = None
    else:
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
            button_key = f"script-runner-run-enabled-{runner.script_key}"
        else:
            disabled = True
            help_text = f"Parameters invalid, please check required ones"
            button_key = f"script-runner-run-disabled-{runner.script_key}"
        
    if sidebar: 
        button_key += "-sidebar"
    return st.button(":material/mode_off_on:", key=button_key , 
                    help = help_text , disabled = disabled , 
                    on_click = SC.click_script_runner_run , args = (runner, params))

def show_main_buttons(runner : ScriptRunner):
    with st.session_state['box-main-button']:
        cols = st.columns(2 , vertical_alignment = "center")
        with cols[0]:
            run_button_button(runner , sidebar = False)
        with cols[1]:
            key = 'current-script-latest-task-button'
            if key not in st.session_state:
                st.session_state[key] = st.empty()
            with st.session_state[key]:
                item = SC.get_latest_task_item(runner.script_key)
                if item is None:
                    st.button(":material/slideshow:", key=f"{key}-disabled-init" , 
                              help = "Please Run a Task of This Script First" , disabled = True)
                else:
                    if st.button(":material/slideshow:", key=f"{key}-enable-{item.id}" , 
                                help = f":blue[**Show Latest Task of This Script**]: {item.id}" , 
                                on_click = SC.click_show_complete_report , args = (item,) ,
                                disabled = False):
                        st.switch_page(runs_page_url(item.script_key))
            
def show_sidebar_buttons(runner : ScriptRunner | None = None):
    if button_placeholder := st.session_state.get('sidebar-runner-button' , None):
        with button_placeholder: 
            run_button_button(runner , sidebar = True)

@clear_and_show
def show_report_main(runner : ScriptRunner):
    """show complete report"""
    item = SC.get_task_item(SC.current_task_item)
    wkey = 'script-task-report'
    header = 'Task Report'
    icon = ':material/overview:'
    if item is None or not item.belong_to(runner):
        help = "Empty! No Task Report is Selected for This Script."
    else:
        help = f"Task Report for {item.id}."
    placeholder = SC.call_report_placeholder()
    with placeholder:
        if item is None or not item.belong_to(runner) : 
            return
        status_text = f'{header}: {item.status_state.title()}'
        status_color = 'green' if item.status_state == 'complete' else 'red' if item.status_state == 'error' else 'blue'
        with expander_subheader(wkey , status_text , icon , True , help = help , status = True , color = status_color):
            if item is None or not item.belong_to(runner): 
                return
            
            start_as_unfinished = item.status_state == 'running'
            with st.expander(":rainbow[:material/build:] **Command Details**", expanded=False):
                st.code(item.cmd , wrap_lines=True)

            st.success(f'{item.running_str} Started!' , icon = ":material/add_task:")

            df_placeholder = st.empty()
            col_config = {
                'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                'Value': st.column_config.TextColumn(width="large", help='Value of the item')
            }
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(info_type = 'enter') , row_height = 20 , column_config = col_config)

            SC.wait_until_completion(item)
            item.refresh()
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(info_type = 'enter') , row_height = 20 , column_config = col_config)

            if item.status == 'error':
                st.error(f'{item.running_str} has Error!' , icon = ":material/error:")
            else:
                st.success(f'{item.running_str} Completed!' , icon = ":material/trophy:")

            exit_info_list = item.info_list(info_type = 'exit' , sep_exit_files = False)
            with st.expander(f":rainbow[:material/fact_check:] **Exit Information**", expanded=True).container(key = 'detail-exit-info-container'):
                for name , value in exit_info_list:
                    if name.lower() == 'exit files':
                        continue
                    st.badge(f"**{name}**" , color = "blue")
                    for s in value.split('\n'):
                        st.write(ColoredText(s))
                    st.write('')

            if item.exit_files:
                st.success(f'{item.running_str} Has {len(item.exit_files)} Exit File(s)!' , icon = ":material/preview:")
                #if SC.running_report_init and len(item.exit_files) == 1:
                #    SC.running_report_file_previewer = Path(item.exit_files[0]).absolute()

                with st.expander(f":rainbow[:material/preview:] **Exit Files**", expanded=True):
                    for i, file in enumerate(item.exit_files):
                        path = Path(file).absolute()
                        col0 , col1, col2 = st.columns([9 , 0.5, 0.5] , vertical_alignment = "center")
                        with col0:
                            if st.button(path.name, key= f"exit-file-open-{path}", 
                                         icon = ":material/open_in_new:" , 
                                         help = f":blue[**Open**]: {path}"):
                                direclty_open_file(path)

                        with col1:
                            preview_key = f"exit-file-preview-{i}" if path != SC.running_report_file_previewer else f"exit-file-preview-select-{i}"
                            st.button(":material/preview:", key=preview_key ,
                                      help = f":blue[**Preview**]: {path}" ,
                                      on_click = SC.click_file_preview , args = (path,))

                        with col2.container(key = f"exit-file-download-{path}"):
                            with open(path, 'rb') as f:
                                if st.download_button(
                                    ':material/download:', 
                                    data=f.read(),
                                    file_name=str(path),
                                    key = f"download-{path}",
                                    help = f":blue[**Download**]: {path}",
                                    on_click=SC.click_file_download , args = (path,)
                                ):
                                    pass
                    
                    previewer = FilePreviewer(SC.running_report_file_previewer)
                    previewer.preview()

    if start_as_unfinished:
        st.rerun()
    else:
        SC.running_report_init = False

def direclty_open_file(path : Path | None = None):
    if path is None:
        return
    path = path.absolute()
    pdf_path = str(path)
    try:
        # Check if the file exists
        if os.path.exists(pdf_path):
            # Use platform-specific commands to open the file
            if MACHINE.is_windows:  
                os.startfile(pdf_path) # type: ignore
            elif MACHINE.is_linux or MACHINE.is_macos: 
                subprocess.run(['open', pdf_path])
            else:
                raise ValueError(f'Unsupported platform: {MACHINE.system_name}')
        else:
            st.error("The file was not found.")
    except Exception as e:
        st.error(f"Could not open the file: {e}")