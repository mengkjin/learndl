import streamlit as st

from typing import Any, Literal, Callable
from pathlib import Path

from src_app.backend import (
    ScriptRunner , ScriptParamInput
)

from src_app.frontend.frontend import (
    FilePreviewer , YAMLFileEditor , ColoredText
)

from .control import SC

def show_script_detail(script_key : str):
    """show main part"""
    runner = SC.script_runners[script_key] if script_key in SC.script_runners else None
    if runner is None:
        st.header(":material/code: No Script Selected")
        return
    else:
        st.header(f":material/code: {runner.script_key}")
        st.warning(f'Description: {runner.content}' , icon = ":material/info:")   
    
    show_script_task_selector(runner)
    show_param_settings(runner)
    show_report_main(runner)

def show_script_task_selector(runner : ScriptRunner , selector_type : Literal['dropdown' , 'expander'] = 'dropdown'):
    if todo := runner.header.todo:
        st.info(f":material/pending_actions: {todo}")

    queue = SC.task_queue.filter(file = [runner.path.path])
    if not queue: return
    item_ids = list(queue.keys())
    
    if selector_type == 'dropdown':
        options = {item.id : " ".join([item.plain_icon, "." ,
                             item.button_str, 
                             f"--ID" , str(item.time_id), 
                             f"--Status" , item.status.title(), 
                             f"--Dur" , str(item.duration_str)]) 
                   for item in queue.values()}
        if SC.choose_task_item is not None and SC.choose_task_item in item_ids:
            index = item_ids.index(SC.choose_task_item)
        else:
            index = None
        st.selectbox("Choose Task Item from Queue", 
                     options = item_ids, 
                     index = index,
                     format_func = lambda x: options[x],
                     key = f"choose-item-selectbox" , 
                     help = "Choose a Task Item from Filtered Queue" ,
                     on_change = SC.click_choose_item_selectbox)
    elif selector_type == 'expander':
        expander = st.expander("Choose Task Item from Queue", expanded = False , icon = ":material/checklist:")
        with expander:
            for item in queue.values():
                col0 , col1 = expander.columns([14, 1] , gap = "small" , vertical_alignment = "center")
                with col0:
                    button_key = f"choose-item-select-{item.id}" if SC.choose_task_item != item.id else f"choose-item-selected-{item.id}"
                    info_text = f"--ID {item.time_id} --Status {item.status.title()} --Dur {item.duration_str}"
                    st.button(f"{item.icon} {item.button_str} {info_text}", 
                                key=button_key , 
                                use_container_width=True , on_click = SC.click_item_choose_select , args = (item,))
                with col1.container(key = f"choose-item-remover-{item.id}"):
                    st.button(":material/cancel:", key = f"choose-item-remover-button-{item.id}",
                                help="Remove/Terminate", 
                                on_click = SC.click_item_choose_remove , args = (item,))

    if SC.choose_task_item:
        st.success(f"Task Item {SC.choose_task_item} chosen" , icon = ":material/check_circle:")

def show_param_settings(runner : ScriptRunner):
    if runner.disabled:
        st.error(f":material/disabled_by_default: This script is disabled")
        return
    with st.container(key = f"script-setting-container-{runner.script_key}"):
        param_inputs = runner.header.get_param_inputs()
        settings_col , collapse_col = st.columns([1, 1] , vertical_alignment = "center")
        with settings_col:
            if not param_inputs:
                st.info("**No parameter settings**" , icon = ":material/settings:")
            else:
                st.info("**Parameter Settings**" , icon = ":material/settings:")

        with collapse_col:
            st.button(":material/close:", key=f"script-setting-classic-remover-{runner.script_key}", 
                      help="Collapse", type="secondary" ,
                      on_click = SC.click_script_runner_expand , args = (runner,))                
        
    params = ParamInputsForm(runner).init_param_inputs('customized').param_values
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
    if SC.ready_to_go(runner):
        help_text = f"Parameters valid, run {runner.script_key}"
        button_key = f"script-runner-run-enabled-{runner.script_key}"
    else:
        help_text = f"Parameters invalid, please check required ones"
        button_key = f"script-runner-run-disabled-{runner.script_key}"
    st.button(":material/mode_off_on:", key=button_key , 
            help = help_text , disabled = not SC.ready_to_go(runner) , 
            on_click = SC.click_script_runner_run , args = (runner,params))

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
    if item is None: return
    if not item.belong_to(runner): return

    status_text = f'Running Report {item.status_state.title()}'
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

            script_str = f"Script [{item.format_path}] ({item.time_str()}) (PID: {item.pid})"
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
