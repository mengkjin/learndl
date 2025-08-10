import streamlit as st

from typing import Any, Literal, Callable
from pathlib import Path

from src_app.backend import (
    ScriptRunner , ScriptParamInput
)

from src_app.frontend.frontend import (
    FilePreviewer , YAMLFileEditor , ColoredText
)

from .control import SC , set_current_page

def change_num_per_page():
    SC.choose_task_item = None
def on_first_page(max_page : int):
    if st.session_state.get('choose-task-page') == 1: return
    #SC.choose_task_item = None
    st.session_state['choose-task-page'] = 1
def on_last_page(max_page : int):
    if st.session_state.get('choose-task-page') == max_page: return
    #SC.choose_task_item = None
    st.session_state['choose-task-page'] = max_page
def on_prev_page(max_page : int):
    if st.session_state.get('choose-task-page') == 1: return
    #SC.choose_task_item = None
    st.session_state['choose-task-page'] = max((st.session_state.get('choose-task-page') or 1) - 1, 1)
def on_next_page(max_page : int):
    if st.session_state.get('choose-task-page') == max_page: return
    #SC.choose_task_item = None
    st.session_state['choose-task-page'] = (st.session_state.get('choose-task-page') or 1) + 1

def show_script_detail(script_key : str):
    """show main part"""
    set_current_page(script_key)
    runner = SC.script_runners[script_key] 
    page = SC.script_pages[script_key]
    if runner is None:
        raise ValueError(f"Script {script_key} not found in SC.script_runners")
    
    header , button = st.columns([18, 2] , gap = "small" , vertical_alignment = "top") 
    with header:
        st.header(f"{page['icon']} {runner.script_key}")
        st.warning(f'**Description: {runner.content.title()}**' , icon = ":material/info:") 
        if todo := runner.header.todo:
            st.info(f"**Todo: {todo.title()}**" , icon = ":material/pending_actions:")  
    

    show_script_task_selector(runner)
    show_param_settings(runner)
    # show_run_button_main()
    show_report_main(runner)
    with button:
        show_run_button_main()
    show_run_button_sidebar()

def show_script_task_selector(runner : ScriptRunner):
    """show script task selector"""
    script_queue = SC.task_queue.filter(file = [runner.path.path])
    if not script_queue: 
        st.subheader(f":blue[:material/history: No Historical Task]" ,
                     help = "No historical task for this script")
        return
    st.subheader(f":blue[:material/history: Choose Historical Task]" ,
                 help = "Choose any task item from the expander below")
    
    format_dict = {item.id : " ".join([
        f"{i+1}." ,
        item.plain_icon, 
        "." ,
        item.button_str, 
        f"--ID {item.time_id : <5}" ,
        f"--Status {item.status.title() : >10}" ,
        f"--Source {item.source.title() : >10}" ,
        f"--Dur {item.duration_str : >10}"
    ]).strip() for i, item in enumerate(script_queue.values())}
    with st.expander(":material/checklist: Choose Task Item", expanded = False):
        status_options = ["All" , "Running" , "Complete" , "Error"]
        source_options = ["All" , "Py" , "App" ,"Bash" , "Other"]
        status = st.radio(":gray-badge[**Running Status**]" , status_options , key = "task-filter-status", horizontal = True)
        source = st.radio(":gray-badge[**Script Source**]" , source_options , key = "task-filter-source", horizontal = True)
        queue = SC.task_queue.filter(status = status , source = source , queue = script_queue)
        st.caption(f":rainbow[:material/bar_chart:] {SC.task_queue.status_message(queue)}")
        st.caption(f":rainbow[:material/bar_chart:] {SC.task_queue.source_message(queue)}")
        
        item_ids = list(queue.keys())
        if SC.choose_task_item is not None and SC.choose_task_item in item_ids:
            choose_index = item_ids.index(SC.choose_task_item)
        else:
            choose_index = None
        
        with st.container(key = f"choose-task-num-per-page-container"):
            cols = st.columns([1, 1 , 2] , vertical_alignment = "center")
            cols[0].info('**Select Number of Task Items per Page**')
            num_per_page = cols[1].selectbox('select num per page', [2 ,10 , 50 , 100], format_func = lambda x: f'{x} Tasks/Page', 
                                index = 0 , key = f"choose-task-item-num-per-page" , label_visibility = 'collapsed' ,
                                on_change = change_num_per_page)
            
        with st.container(key = f"choose-task-page-container"):
            max_page = (len(item_ids) - 1) // num_per_page + 1
            page_options = list(range(1, max_page + 1))
            index_page = choose_index // num_per_page if choose_index is not None else 0
            page_cols = st.columns([7, 1, 1, 3, 1, 1 , 14] , vertical_alignment = "center")
            page_cols[0].info('**Select Page**')
            page_cols[1].button(":material/first_page:", key = f"choose-task-page-first", on_click = on_first_page , args = (max_page,))
            page_cols[2].button(":material/chevron_left:", key = f"choose-task-page-prev", on_click = on_prev_page , args = (max_page,))
            page_cols[3].selectbox('select page', page_options, key = f"choose-task-page" , index = index_page , 
                                   placeholder = f'Page #',
                                   format_func = lambda x: f'Page {x}', label_visibility = 'collapsed')
            page_cols[4].button(":material/chevron_right:", key = f"choose-task-page-next", on_click = on_next_page , args = (max_page,))
            page_cols[5].button(":material/last_page:", key = f"choose-task-page-last", on_click = on_last_page , args = (max_page,))
        
        with st.container(key = f"choose-task-item-container"):
            current_page = st.session_state.get('choose-task-page') or 1
            item_options = item_ids[(current_page - 1) * num_per_page : current_page * num_per_page]
            item_index = (choose_index % num_per_page) if choose_index is not None else None
            st.info('**Choose Task Item from Queue**')
            st.selectbox("Choose Task Item from Queue", 
                        options = item_options, 
                        index = item_index,
                        format_func = lambda x: format_dict[x],
                        key = f"choose-item-selectbox" , 
                        help = "Choose a Task Item from Filtered Queue" ,
                        placeholder = "Choose a Task Item from Filtered Queue",
                        on_change = SC.click_choose_item_selectbox , 
                        label_visibility = 'collapsed')
            
        
        if SC.choose_task_item:
            st.success(f"Task Item {SC.choose_task_item} chosen" , icon = ":material/check_circle:")

def show_param_settings(runner : ScriptRunner):
    if runner.disabled:
        st.error(f":material/disabled_by_default: This script is disabled")
        return
    param_inputs = runner.header.get_param_inputs()
    if not param_inputs:
        st.subheader(f":blue[:material/settings: No Parameters]" ,
                     help = "No parameter is required for this script")
    else:
        st.subheader(f":blue[:material/settings: Parameter Settings]" ,
                    help = "Input parameters for this script in the expander below , mind the required ones")
            
    with st.expander(":material/settings: Parameter Settings", expanded = True):
        param_input_form = ParamInputsForm(runner).init_param_inputs('customized')
        SC.param_inputs_form = param_input_form
        params = param_input_form.param_values
        if runner.header.file_editor:
            with st.expander(runner.header.file_editor.get('name', 'File Editor') , expanded = False , icon = ":material/edit_document:"):
                path = runner.header.file_editor['path'].format(**params)
                file_editor = YAMLFileEditor('param-settings-file-editor', 
                                            file_root=path , file_input=False , 
                                            height = runner.header.file_editor.get('height'))
                file_editor.show_yaml_editor()
        if runner.header.file_previewer:
            with st.expander(runner.header.file_previewer.get('name', 'File Previewer') , expanded = False , icon = ":material/file_present:"):
                path = runner.header.file_previewer['path'].format(**params)
                file_previewer = FilePreviewer(path , height = runner.header.file_previewer.get('height'))
                file_previewer.preview()
    
def run_button_header(sidebar = False):
    runner = SC.current_script_runner
    if runner is None:
        help_text = f"Please Choose a Script to Run First"
    elif SC.ready_to_go(runner):
        help_text = f"Parameters valid, run {runner.script_key}"
    else:
        help_text = f"Parameters invalid, please check required ones"
        
    st.subheader(f":blue[:material/run_circle: Run Script]" , help = help_text)

def run_button_button(sidebar = False):
    runner = SC.current_script_runner
    if runner is None:
        disabled = True
        help_text = f"Please Choose a Script to Run First"
        button_key = f"script-runner-run-disabled-not-selected"
        params = None

    else:
        if SC.param_inputs_form is None:
            raise ValueError("ParamInputsForm is not initialized")
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
        
        if SC.ready_to_go(runner):
            disabled = False
            help_text = f"Parameters valid, run {runner.script_key}"
            button_key = f"script-runner-run-enabled-{runner.script_key}"
        else:
            disabled = True
            help_text = f"Parameters invalid, please check required ones"
            button_key = f"script-runner-run-disabled-{runner.script_key}"
        
    if sidebar: button_key += "-sidebar"
    st.button(":material/mode_off_on:", key=button_key , 
            help = help_text , disabled = disabled , 
            on_click = SC.click_script_runner_run , args = (runner, params))

def show_run_button_main():
    # run_button_header(sidebar = False)
    run_button_button(sidebar = False)

def show_run_button_sidebar():
    #if header_placeholder := st.session_state.get('sidebar-runner-header' , None):
    #    with header_placeholder: run_button_header(sidebar = True)
    if button_placeholder := st.session_state.get('sidebar-runner-button' , None):
        with button_placeholder: run_button_button(sidebar = True)
    
class ParamInputsForm:
    def __init__(self , runner : ScriptRunner):
        self.runner = runner
        self.param_list = [self.WidgetParamInput(runner, p) for p in runner.header.get_param_inputs()]
        self.errors = []

    def init_param_inputs(self , type : Literal['customized', 'form'] = 'customized'):
        trigger_item = SC.task_queue.get(SC.script_runner_trigger_item)
        cmd = trigger_item.cmd if trigger_item is not None else ''
        if type == 'customized':
            self.init_customized_container(cmd)
        elif type == 'form':
            self.init_form(cmd)
        else:
            raise ValueError(f"Invalid param inputs type: {type}")
        SC.script_runner_trigger_item = None
        return self

    class WidgetParamInput(ScriptParamInput):
        def __init__(self , runner : ScriptRunner , param : ScriptParamInput):
            super().__init__(**param.as_dict())
            self._runner = runner
            self._param = param
            self.widget_key = self.get_widget_key(runner, param)
            self.transform = self.value_transform(param)

        @property
        def script_key(self):
            return self._runner.script_key

        @property
        def raw_value(self) -> Any:
            return st.session_state[self.widget_key]
        
        @property
        def param_value(self):
            return self.transform(self.raw_value)
        
        def is_valid(self):
            return self._param.is_valid(self.param_value)
        
        def error_message(self):
            return self._param.error_message(self.param_value)
        
        @classmethod
        def get_widget_key(cls , runner : ScriptRunner , param : ScriptParamInput):
            return f"script-param-{runner.script_key}-{param.name}"
        
        @classmethod
        def value_transform(cls , param : ScriptParamInput):
            ptype = param.ptype
            if isinstance(ptype, list):
                options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
                return cls.raw_option([None] + ptype, options)
            elif ptype == str:
                return lambda x: (x.strip() if x is not None else None)
            elif ptype == bool:
                return lambda x: None if x is None or x == 'Choose an option' else bool(x)
            elif ptype == int:
                return lambda x: None if x is None else int(x)
            elif ptype == float:
                return lambda x: None if x is None else float(x)
            else:
                raise ValueError(f"Unsupported param type: {ptype}")

        @classmethod
        def raw_option(cls , raw_values : list[Any], alt_values : list[Any]):
            def wrapper(alt : Any):
                """get index of value in options"""
                if alt is None or alt == '': alt = 'Choose an option'
                assert alt in alt_values , f"Invalid option '{alt}' in list {alt_values}"
                raw = raw_values[alt_values.index(alt)] if alt is not None else None
                return raw
            return wrapper
        
        def on_change(self):
            if self.script_key not in SC.script_params_cache:
                SC.script_params_cache[self.script_key] = {
                    'raw': {},
                    'value': {},
                    'valid': {}
                }
            SC.script_params_cache[self.script_key]['raw'][self.name] = self.raw_value
            SC.script_params_cache[self.script_key]['value'][self.name] = self.param_value
            SC.script_params_cache[self.script_key]['valid'][self.name] = self.is_valid()

    def cmd_to_param_values(self , cmd : str = ''):
        param_values = {}
        if not cmd or not str(self.runner.path.path) in cmd: return param_values
        main_str = [s for s in cmd.split(";") if str(self.runner.path.path) in s][0]
        param_str = ''.join(main_str.split(str(self.runner.path.path))[1:]).strip()
        
        for param_str in param_str.split('--'):
            if not param_str: continue
            param_name , param_value = param_str.split(' ' , 1)
            value = param_value.strip()
            if value == 'True': value = True
            elif value == 'False': value = False
            elif value == 'None': value = None
            param_values[param_name] = value
        return param_values

    def init_customized_container(self , cmd : str = '' , num_cols : int = 3):
        num_cols = min(num_cols, len(self.param_list))
        cmd_param_values = self.cmd_to_param_values(cmd)
        for i, wp in enumerate(self.param_list):
            if i % num_cols == 0:
                param_cols = st.columns(num_cols)
            with param_cols[i % num_cols]:
                self.get_widget(
                    runner=self.runner, param=wp, value = cmd_param_values.get(wp.name) ,
                    on_change=self.on_widget_change, args=(wp,))
                self.on_widget_change(wp)
                if err_msg := wp.error_message():
                    st.error(err_msg , icon = ":material/error:")
        return self
    
    def init_form(self , cmd : str):
        cmd_param_values = self.cmd_to_param_values(cmd)
        with st.form(f"ParamInputsForm-{self.runner.script_key}" , clear_on_submit = False):
            for param in self.param_list:
                self.get_widget(self.runner, param, value = cmd_param_values.get(param.name))

            if st.form_submit_button(
                "Submit" ,
                help = "Submit Parameters to Run Script" ,
            ):
                self.submit()

        return self

    @property
    def param_values(self):
        return {wp.name: wp.param_value for wp in self.param_list}

    def validate(self):
        self.errors = []
        for wp in self.param_list:
            if err_msg := wp.error_message():
                self.errors.append(err_msg)
                
        return len(self.errors) == 0

    def submit(self):
        for wp in self.param_list: wp.on_change()
            
        if not self.validate():
            for err_msg in self.errors:
                st.error(err_msg , icon = ":material/error:")
    
    def process(self):
        ...

    @classmethod
    def get_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                   value : Any | None = None ,
                   on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        
        if isinstance(ptype, list):
            func = cls.list_widget
        elif ptype == str:
            func = cls.text_widget
        elif ptype == bool:
            func = cls.bool_widget
        elif ptype == int:
            func = cls.int_widget
        elif ptype == float:
            func = cls.float_widget
        else:
            raise ValueError(f"Unsupported param type: {ptype}")
        
        return func(runner, param, value, on_change = on_change, args = args, kwargs = kwargs)

    @classmethod
    def value_transform(cls , param : ScriptParamInput):
        return cls.WidgetParamInput.value_transform(param)
    
    @classmethod
    def get_title(cls , param : ScriptParamInput):
        return f':red[:material/asterisk: **{param.title}**]' if param.required else f'**{param.title}**'
    
    @classmethod
    def get_widget_key(cls , runner : ScriptRunner , param : ScriptParamInput):
        return f"script-param-{runner.script_key}-{param.name}"

    @classmethod
    def get_default_value(cls , runner : ScriptRunner , param : ScriptParamInput , value : Any | None = None):
        widget_key = cls.get_widget_key(runner, param)
        if value is not None:
            default_value = f'{param.prefix}{value}'
        else:
            default_value = SC.script_params_cache.get(runner.script_key, {}).get('raw', {}).get(param.name, param.default)
        if default_value is None:
            default_value = st.session_state[widget_key] if widget_key in st.session_state else param.default
        return default_value
    
    @classmethod
    def get_widget_value(cls , runner : ScriptRunner , param : ScriptParamInput):
        widget_key = cls.get_widget_key(runner, param)
        return st.session_state[widget_key]

    @classmethod
    def list_widget(cls , runner : ScriptRunner , param : ScriptParamInput , 
                    value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert isinstance(ptype, list) , f"Param {param.name} is not a list"
        
        widget_key = cls.get_widget_key(runner, param)
        default_value = cls.get_default_value(runner, param , value)
        title = cls.get_title(param)
        options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
        if default_value is not None and default_value not in options:
            return st.text_input(
                title,
                value=None if default_value is None else str(default_value),
                placeholder=param.placeholder ,
                key=widget_key ,
                **kwargs
            )
        else:
            return st.selectbox(
                title,
                options,
                index=0 if default_value is None else options.index(default_value),
                key=widget_key ,
                **kwargs
            )
    
    @classmethod
    def text_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                    value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == str , f"Param {param.name} is not a string"
        widget_key = cls.get_widget_key(runner, param)
        default_value = cls.get_default_value(runner, param , value)
        title = cls.get_title(param)
        return st.text_input(
            title,
            value=None if default_value is None else str(default_value),
            placeholder=param.placeholder ,
            key=widget_key ,
            **kwargs
        )
    
    @classmethod
    def bool_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                    value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == bool , f"Param {param.name} is not a boolean"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param , value)
        return st.selectbox(
            title,
            ['Choose an option', True, False],
            index=0 if default_value is None else 2-bool(default_value),    
            key=widget_key ,
            **kwargs
        )
    
    @classmethod
    def int_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                   value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == int , f"Param {param.name} is not an integer"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param , value)
        return st.number_input(
            title,
            value=None if default_value is None else int(default_value),
            min_value=param.min,
            max_value=param.max,
            placeholder=param.placeholder,
            key=widget_key ,
            **kwargs
        )
    
    @classmethod
    def float_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                     value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == float , f"Param {param.name} is not a float"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param , value)
        return st.number_input(
            title,
            value=None if default_value is None else float(default_value),
            min_value=param.min,
            max_value=param.max,
            step=0.1,
            placeholder=param.placeholder,
            key=widget_key ,
            **kwargs
        )

    @classmethod
    def get_form_errors(cls):
        return st.session_state.form_errors
    
    @classmethod
    def on_widget_change(cls , wp : WidgetParamInput):
        wp.on_change()
        
def show_report_main(runner : ScriptRunner):
    """show complete report"""
    item = SC.task_queue.get(SC.current_task_item)
    if item is None or not item.belong_to(runner): 
        st.subheader(f":blue[:material/overview: Task Full Report (Empty)]" ,
                     help = f"No task has been selected for {runner.script_key}")
        return
    
    st.subheader(f":blue[:material/overview: Task Full Report]" ,
                 help = f"Task report for {item.id}")
    
    status_text = f'Status: {item.status_state.title()}'
    if item.status == 'complete':
        status_text = f':green[**{status_text}**]'
    elif item.status == 'error':
        status_text = f':red[**{status_text}**]'
    else:
        status_text = f':orange[**{status_text}**]'
    status_placeholder = st.empty()
    status = status_placeholder.status(status_text , state = item.status_state , expanded = True)
    
    start_as_unfinished = item.status not in ['complete' , 'error']
    with status_placeholder:
        if not SC.running_report_main_cleared:
            st.write('')
            SC.running_report_main_cleared = True
            SC.running_report_file_previewer = None
            st.rerun()

        with status:
            with st.expander(":rainbow[:material/build:] **Command Details**", expanded=False):
                st.code(item.cmd , wrap_lines=True)

            script_str = f"Script [{item.format_path}] --Time: {item.time_str()} --ID: {item.time_id} --Source: {item.source.title()} --PID: {item.pid}"
            st.success(f'{script_str} started' , icon = ":material/add_task:")

            df_placeholder = st.empty()
            col_config = {
                'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                'Value': st.column_config.TextColumn(width="large", help='Value of the item')
            }
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(info_type = 'enter') , row_height = 20 , column_config = col_config)

            SC.wait_for_complete(item)
            item.refresh()
            with df_placeholder.expander(":rainbow[:material/data_table:] **Running Information**", expanded=True):
                st.dataframe(item.dataframe(info_type = 'enter') , row_height = 20 , column_config = col_config)

            if item.status == 'error':
                st.error(f'{script_str} has error' , icon = ":material/error:")
            else:
                st.success(f'{script_str} Completed' , icon = ":material/trophy:")

            exit_info_list = item.info_list(info_type = 'exit')
            
            with st.expander(f":rainbow[:material/fact_check:] **Exit Information**", expanded=True):
                for name , value in exit_info_list:
                    st.badge(f"**{name}**" , color = "blue")
                    for s in value.split('\n'):
                        st.write(ColoredText(s))
                    st.markdown('')

            if item.exit_files:
                with st.expander(f":rainbow[:material/file_present:] **File Previewer**", expanded=True):
                    for file in item.exit_files:
                        path = Path(file).absolute()
                        preview_key = f"file-preview-{path}" if SC.running_report_file_previewer != path else f"file-preview-selected-{path}"
                        col1, col2 = st.columns([4, 1] , vertical_alignment = "center")
                        with col1:
                            st.button(path.name, key=preview_key , icon = ":material/file_present:" , 
                                      help = f"Preview {path}" ,
                                      on_click = SC.click_file_preview , args = (path,))

                        with col2.container(key = f"file-download-{path}"):
                            with open(path, 'rb') as f:
                                if st.download_button(
                                    ':material/download:', 
                                    data=f.read(),
                                    file_name=str(path),
                                    key = f"download-{path}",
                                    help = f"Download {path}",
                                    on_click=SC.click_file_download , args = (path,)
                                ):
                                    pass

                    previewer = FilePreviewer(SC.running_report_file_previewer)
                    previewer.preview()

    if start_as_unfinished:
        st.rerun()
