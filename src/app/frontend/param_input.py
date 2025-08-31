from typing import Literal , Any , Callable
import streamlit as st

from src.app.backend import ScriptRunner , ScriptParamInput , TaskItem

class ParamInputsForm:
    def __init__(self , runner : ScriptRunner , target : dict[str, Any] , item : TaskItem | None = None):
        self.runner = runner
        self.param_dict = {p.name:self.WidgetParamInput(runner, p) for p in runner.header.get_param_inputs()}
        self.errors = []
        self.trigger_item = item
        self.target = target

    def init_param_inputs(self , type : Literal['customized', 'form'] = 'customized'):
        cmd = self.trigger_item.cmd if self.trigger_item is not None else ''
        if type == 'customized':
            self.init_customized_container(cmd)
        elif type == 'form':
            self.init_form(cmd)
        else:
            raise ValueError(f"Invalid param inputs type: {type}")
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
            return st.session_state.get(self.widget_key, None)
        
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
        
        def on_change(self , target_dict : dict[str, Any]):
            if self.script_key not in target_dict:
                target_dict[self.script_key] = {
                    'raw': {},
                    'value': {},
                    'valid': {}
                }
            target_dict[self.script_key]['raw'][self.name] = self.raw_value
            target_dict[self.script_key]['value'][self.name] = self.param_value
            target_dict[self.script_key]['valid'][self.name] = self.is_valid()

    def cmd_to_param_values(self , cmd : str = ''):
        param_values = {}
        if not cmd or not str(self.runner.path.path) in cmd: return param_values
        main_str = [s for s in cmd.split(";") if str(self.runner.path.path) in s][0]
        param_str = ''.join(main_str.split(str(self.runner.path.path))[1:]).strip()
        
        for pstr in param_str.split('--'):
            if not pstr: continue
            try:
                param_name , param_value = pstr.replace('=' , ' ').split(' ' , 1)
            except Exception as e:
                print(cmd)
                print(pstr)
                print(f"Error parsing param: {pstr} - {e}")
                raise e

            value = param_value.strip()
            if value == 'True': value = True
            elif value == 'False': value = False
            elif value == 'None': value = None
            param_values[param_name] = value
        return param_values

    def init_customized_container(self , cmd : str = '' , num_cols : int = 4):
        num_cols = min(num_cols, len(self.param_dict))
        cmd_param_values = self.cmd_to_param_values(cmd)
        for i, wp in enumerate(self.param_dict.values()):
            if i % num_cols == 0:
                param_cols = st.columns(num_cols)
            with param_cols[i % num_cols]:
                self.get_widget(
                    runner=self.runner, param=wp, 
                    target=self.target,
                    value = cmd_param_values.get(wp.name) ,
                    on_change=self.on_widget_change, args=(wp, self.target))
                if err_msg := wp.error_message():
                    st.error(err_msg , icon = ":material/error:")
        return self
    
    def init_form(self , cmd : str):
        cmd_param_values = self.cmd_to_param_values(cmd)
        with st.form(f"ParamInputsForm-{self.runner.script_key}" , clear_on_submit = False):
            for param in self.param_dict.values():
                self.get_widget(self.runner, param, target=self.target, value = cmd_param_values.get(param.name))

            if st.form_submit_button(
                "Submit" ,
                help = "Submit Parameters to Run Script" ,
            ):
                self.submit()

        return self

    @property
    def param_values(self):
        return {wp.name: wp.param_value for wp in self.param_dict.values()}

    def validate(self):
        self.errors = []
        for wp in self.param_dict.values():
            if err_msg := wp.error_message():
                self.errors.append(err_msg)
                
        return len(self.errors) == 0

    def submit(self):
        for wp in self.param_dict.values(): wp.on_change(self.target)
            
        if not self.validate():
            for err_msg in self.errors:
                st.error(err_msg , icon = ":material/error:")
    
    def process(self):
        ...

    @classmethod
    def get_widget(cls , runner : ScriptRunner , param : ScriptParamInput , target : dict[str, Any],
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
        
        return func(runner, param, target, value, on_change = on_change, args = args, kwargs = kwargs)

    @classmethod
    def value_transform(cls , param : ScriptParamInput):
        return cls.WidgetParamInput.value_transform(param)
    
    @classmethod
    def get_title(cls , param : ScriptParamInput):
        return f':red[:material/asterisk: **{param.title}**]' if param.required else f'**{param.title}**'
    
    @classmethod
    def get_help(cls , param : ScriptParamInput):
        help_texts = []
        if param.required: help_texts.append(f':red[**Required Parameter!**]')
        if param.desc: help_texts.append(f'*{param.desc}*')
        return '\t'.join(help_texts)

    @classmethod
    def get_widget_key(cls , runner : ScriptRunner , param : ScriptParamInput):
        return f"script-param-{runner.script_key}-{param.name}"

    @classmethod
    def get_default_value(cls , runner : ScriptRunner , param : ScriptParamInput , target : dict[str, Any], value : Any | None = None):
        widget_key = cls.get_widget_key(runner, param)
        if value is not None:
            default_value = f'{param.prefix}{value}'
        else:
            default_value = target.get(runner.script_key, {}).get('raw', {}).get(param.name, param.default)
        if default_value is None:
            default_value = st.session_state[widget_key] if widget_key in st.session_state else param.default
        return default_value
    
    @classmethod
    def get_widget_value(cls , runner : ScriptRunner , param : ScriptParamInput):
        widget_key = cls.get_widget_key(runner, param)
        return st.session_state[widget_key]

    @classmethod
    def list_widget(cls , runner : ScriptRunner , param : ScriptParamInput , target : dict[str, Any],
                    value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert isinstance(ptype, list) , f"Param {param.name} is not a list"
        
        widget_key = cls.get_widget_key(runner, param)
        default_value = cls.get_default_value(runner, param , target , value)
        title = cls.get_title(param)
        help = cls.get_help(param)
        options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
        if default_value is not None and default_value not in options:
            return st.text_input(
                title,
                value=None if default_value is None else str(default_value),
                placeholder=param.placeholder ,
                key=widget_key ,
                help=help,
                **kwargs
            )
        else:
            return st.selectbox(
                title,
                options,
                index=0 if default_value is None else options.index(default_value),
                key=widget_key ,
                help=help,
                **kwargs
            )
    
    @classmethod
    def text_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                    target : dict[str, Any],
                    value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == str , f"Param {param.name} is not a string"
        widget_key = cls.get_widget_key(runner, param)
        default_value = cls.get_default_value(runner, param , target , value)
        title = cls.get_title(param)
        help = cls.get_help(param)
        return st.text_input(
            title,
            value=None if default_value is None else str(default_value),
            placeholder=param.placeholder ,
            key=widget_key ,
            help=help,
            **kwargs
        )
    
    @classmethod
    def bool_widget(cls , runner : ScriptRunner , param : ScriptParamInput , 
                    target : dict[str, Any],
                    value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == bool , f"Param {param.name} is not a boolean"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param , target , value)
        help = cls.get_help(param)
        return st.selectbox(
            title,
            ['Choose an option', True, False],
            index=0 if default_value is None else 2-bool(default_value),    
            key=widget_key ,
            help=help,
            **kwargs
        )
    
    @classmethod
    def int_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                   target : dict[str, Any],
                   value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == int , f"Param {param.name} is not an integer"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param , target , value)
        help = cls.get_help(param)
        return st.number_input(
            title,
            value=None if default_value is None else int(default_value),
            min_value=param.min,
            max_value=param.max,
            placeholder=param.placeholder,
            key=widget_key ,
            help=help,
            **kwargs
        )
    
    @classmethod
    def float_widget(cls , runner : ScriptRunner , param : ScriptParamInput ,
                     target : dict[str, Any],
                     value : Any | None = None , **kwargs):
        ptype = param.ptype
        assert ptype == float , f"Param {param.name} is not a float"
        widget_key = cls.get_widget_key(runner, param)
        title = cls.get_title(param)
        default_value = cls.get_default_value(runner, param , target , value)
        help = cls.get_help(param)
        return st.number_input(
            title,
            value=None if default_value is None else float(default_value),
            min_value=param.min,
            max_value=param.max,
            step=0.1,
            placeholder=param.placeholder,
            key=widget_key ,
            help=help,
            **kwargs
        )

    @classmethod
    def get_form_errors(cls):
        return st.session_state.form_errors
    
    @classmethod
    def on_widget_change(cls , wp : WidgetParamInput , target : dict[str, Any]):
        wp.on_change(target)
