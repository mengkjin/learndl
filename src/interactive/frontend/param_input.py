from typing import Literal , Any , Callable
import streamlit as st

from src.proj import Logger
from src.interactive.backend import ScriptRunner , ScriptParamInput , TaskItem

from .param_cache import ParamCache

class WidgetParamInput(ScriptParamInput):
    def __init__(self , runner : ScriptRunner , param : ScriptParamInput):
        super().__init__(**param.as_dict())
        self._runner = runner
        self._param = param
        self.option_to_value = self.option_to_value_transformer(param)
        self.value_to_option = self.value_to_option_transformer(param)

    @property
    def script_key(self):
        return self._runner.script_key

    @property
    def option(self) -> Any:
        return st.session_state.get(self.widget_key, None)
    
    @property
    def param_value(self):
        return self.option_to_value(self.option)

    @property
    def widget_key(self):
        return f"script-param-{self.script_key}-{self.name}"
    
    def is_valid(self):
        return self._param.is_valid(self.param_value)
    
    def error_message(self):
        return self._param.error_message(self.param_value)
    
    @classmethod
    def option_to_value_transformer(cls , param : ScriptParamInput):
        ptype = param.ptype
        if isinstance(ptype, list):
            options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
            values = [None] + ptype
            def wrapper(option : Any):
                """get index of value in options"""
                if option is None or option == '' or option == 'Choose an option': 
                    option = 'Choose an option'
                if option not in options:
                    option = cls.remove_extra_prefix_regex(option, param.prefix, remain_prefix = False)
                assert option in options , f"Invalid option '{option}' in list {options}"
                value = values[options.index(option)]
                return value
            return wrapper
        elif ptype is str:
            return lambda x: (x.strip() if x is not None else None)
        elif ptype is bool:
            return lambda x: None if x is None or x == 'Choose an option' else bool(x)
        elif ptype is int:
            return lambda x: None if x is None else int(x)
        elif ptype is float:
            return lambda x: None if x is None else float(x)
        else:
            raise ValueError(f"Unsupported param type: {ptype}")

    @classmethod
    def value_to_option_transformer(cls , param : ScriptParamInput):
        ptype = param.ptype
        if isinstance(ptype, list):
            options = ['Choose an option'] + [f'{param.prefix}{e}' for e in ptype]
            values = [None] + [str(ptype_e) for ptype_e in ptype]
            def wrapper(value : Any):
                """get index of value in options"""
                if value is None or value == '' or value == 'Choose an option': 
                    value = None
                if value not in values:
                    value = cls.remove_extra_prefix_regex(value, param.prefix , remain_prefix = False)
                assert value in values , f"Invalid value '{value}' in list {values}"
                option = options[values.index(value)]
                return option
            return wrapper
        elif ptype is str:
            return lambda x: (None if x is None or x == '' else x.strip())
        elif ptype is bool:
            return lambda x: 'Choose an option' if x is None or x == 'Choose an option' else bool(x)
        elif ptype is int:
            return lambda x: None if x is None else int(x)
        elif ptype is float:
            return lambda x: None if x is None else float(x)
        else:
            raise ValueError(f"Unsupported param type: {ptype}")

    @classmethod
    def remove_extra_prefix_regex(cls , s : Any, prefix : str , remain_prefix : bool = True):
        s = str(s)
        if not prefix:
            return s
        while s.startswith(prefix):
            s = s.removeprefix(prefix)
        if remain_prefix:
            return f'{prefix}{s}'
        else:    
            return s
    
    def on_change(self , cache : ParamCache):
        cache.set(self.option, self.script_key, 'option', self.name)
        cache.set(self.param_value, self.script_key, 'value', self.name)
        cache.set(self.is_valid(), self.script_key, 'valid', self.name)

    def default_option(self , cache : ParamCache, value : Any | None = None):
        if value is not None:
            default_option = self.value_to_option(value)
        else:
            default_option = None
            if cache.has(self.script_key, 'option', self.name):
                default_option = cache.get(self.script_key, 'option', self.name)
            elif cache.has(self.script_key, 'value', self.name):
                default_option = self.value_to_option(cache.get(self.script_key, 'value', self.name))
            #elif self.widget_key in st.session_state:
            #    default_option = st.session_state.get(self.widget_key, None)
            else:
                default_option = self.value_to_option(self.default)
        return default_option

class ParamInputsForm:
    def __init__(self , runner : ScriptRunner , cache : ParamCache , item : TaskItem | None = None):
        self.runner = runner
        self.param_dict = {p.name:WidgetParamInput(runner, p) for p in runner.header.get_param_inputs()}
        self.errors = []
        self.trigger_item = item
        self.cache = cache

    def init_param_inputs(self , type : Literal['customized', 'form'] = 'customized' , cmd : str | None = None):
        self.type = type
        if cmd is None:
            cmd = self.trigger_item.cmd if self.trigger_item is not None else ''
        else:
            cmd = cmd
        if type == 'customized':
            self.init_customized_container(cmd)
        elif type == 'form':
            self.init_form(cmd)
        else:
            raise ValueError(f"Invalid param inputs type: {type}")
        return self

    def cmd_to_param_values(self , cmd : str = ''):
        param_values = {}
        if not cmd or str(self.runner.path.path) not in cmd: 
            return param_values
        main_str = [s for s in cmd.split(";") if str(self.runner.path.path) in s][0]
        param_str = ''.join(main_str.split(str(self.runner.path.path))[1:]).strip()
        
        for pstr in param_str.split('--'):
            if not pstr: 
                continue
            try:
                param_name , param_value = pstr.replace('=' , ' ').split(' ' , 1)
            except Exception as e:
                Logger.stdout(cmd)
                Logger.stdout(pstr)
                Logger.error(f"Error parsing param: {pstr} - {e}")
                raise e

            value = param_value.strip()
            if value == 'True': 
                value = True
            elif value == 'False': 
                value = False
            elif value == 'None': 
                value = None
            param_values[param_name] = value
        return param_values

    def init_customized_container(self , cmd : str = '' , num_cols : int = 4):
        num_cols = min(num_cols, len(self.param_dict))
        cmd_param_values = self.cmd_to_param_values(cmd)
        for i, wp in enumerate[WidgetParamInput](self.param_dict.values()):
            if i % num_cols == 0:
                param_cols = st.columns(num_cols)
            with param_cols[i % num_cols]:
                self.get_widget(
                    runner=self.runner, param=wp, 
                    cache=self.cache,
                    value = cmd_param_values.get(wp.name) ,
                    on_change=self.on_widget_change, args=(wp, self.cache))
                if err_msg := wp.error_message():
                    st.error(err_msg , icon = ":material/error:")
        return self
    
    def init_form(self , cmd : str):
        cmd_param_values = self.cmd_to_param_values(cmd)
        with st.form(f"ParamInputsForm-{self.runner.script_key}" , clear_on_submit = False):
            for param in self.param_dict.values():
                self.get_widget(self.runner, param, cache=self.cache, value = cmd_param_values.get(param.name))

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
        for wp in self.param_dict.values(): 
            wp.on_change(self.cache)
            
        if not self.validate():
            for err_msg in self.errors:
                st.error(err_msg , icon = ":material/error:")
    
    def process(self):
        ...

    @classmethod
    def get_widget(cls , runner : ScriptRunner , param : WidgetParamInput , cache : ParamCache,
                   value : Any | None = None ,
                   on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None):
        ptype = param.ptype
        
        if isinstance(ptype, list):
            func = cls.list_widget
        elif ptype is str:
            func = cls.text_widget
        elif ptype is bool:
            func = cls.bool_widget
        elif ptype is int:
            func = cls.int_widget
        elif ptype is float:
            func = cls.float_widget
        else:
            raise ValueError(f"Unsupported param type: {ptype}")
        
        return func(runner, param, cache, value, on_change = on_change, args = args, kwargs = kwargs)
    
    @classmethod
    def get_title(cls , param : WidgetParamInput):
        return f':red[:material/asterisk: **{param.title}**]' if param.required else f'**{param.title}**'
    
    @classmethod
    def get_help(cls , param : WidgetParamInput):
        help_texts = []
        if param.required: 
            help_texts.append(f':red[**Required Parameter!**]')
        if param.desc: 
            help_texts.append(f'*{param.desc}*')
        return '\t'.join(help_texts)

    @classmethod
    def list_widget(cls , runner : ScriptRunner , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs):
        assert isinstance(param.ptype, list) , f"Param {param.name} is not a list"
        
        default_option = param.default_option(cache, value)
        title = cls.get_title(param)
        help = cls.get_help(param)
        options = ['Choose an option'] + [f'{param.prefix}{e}' for e in param.ptype]
        if default_option is not None and default_option not in options:
            return st.text_input(
                title,
                value=None if default_option is None else str(default_option),
                placeholder=param.placeholder ,
                key=param.widget_key ,
                help=help,
                **kwargs
            )
        else:
            return st.selectbox(
                title,
                options,
                index=0 if default_option is None else options.index(default_option),
                key=param.widget_key ,
                help=help,
                **kwargs
            )
    
    @classmethod
    def text_widget(cls , runner : ScriptRunner , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs):
        assert param.ptype is str , f"Param {param.name} is not a string"
        default_option = param.default_option(cache, value)
        title = cls.get_title(param)
        help = cls.get_help(param)
        return st.text_input(
            title,
            value=None if default_option is None else str(default_option),
            placeholder=param.placeholder ,
            key=param.widget_key ,
            help=help,
            **kwargs
        )
    
    @classmethod
    def bool_widget(cls , runner : ScriptRunner , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs):
        assert param.ptype is bool , f"Param {param.name} is not a boolean"
        title = cls.get_title(param)
        default_option = param.default_option(cache, value)
        help = cls.get_help(param)
        return st.selectbox(
            title,
            ['Choose an option', True, False],
            index=0 if default_option is None else 2-bool(default_option),    
            key=param.widget_key ,
            help=help,
            **kwargs
        )
    
    @classmethod
    def int_widget(cls , runner : ScriptRunner , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs):
        assert param.ptype is int , f"Param {param.name} is not an integer"
        title = cls.get_title(param)
        default_option = param.default_option(cache, value)
        help = cls.get_help(param)
        return st.number_input(
            title,
            value=None if default_option is None else int(default_option),
            min_value=param.min,
            max_value=param.max,
            placeholder=param.placeholder,
            key=param.widget_key ,
            help=help,
            **kwargs
        )
    
    @classmethod
    def float_widget(cls , runner : ScriptRunner , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs):
        assert param.ptype is float , f"Param {param.name} is not a float"
        title = cls.get_title(param)
        default_option = param.default_option(cache, value)
        help = cls.get_help(param)
        return st.number_input(
            title,
            value=None if default_option is None else float(default_option),
            min_value=param.min,
            max_value=param.max,
            step=0.1,
            placeholder=param.placeholder,
            key=param.widget_key ,
            help=help,
            **kwargs
        )

    @classmethod
    def get_form_errors(cls):
        return st.session_state.form_errors
    
    @classmethod
    def on_widget_change(cls , wp : WidgetParamInput , cache : ParamCache):
        wp.on_change(cache)
