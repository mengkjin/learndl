"""
Streamlit widget layer for script parameter input.

Provides :class:`WidgetParamInput` (a Streamlit-aware extension of
:class:`~src.interactive.backend.ScriptParamInput`) and :class:`ParamInputsForm`,
which renders a complete parameter form for a given script and manages the
option ↔ value transformation pipeline.
"""
from __future__ import annotations
from typing import Literal , Any , Callable
import streamlit as st
import ast
import re

from dataclasses import dataclass
from src.proj import Logger
from src.proj.util import Options # noqa
from src.interactive.backend import ScriptRunner , ScriptParamInput , TaskItem

class NoCachedValue:
    """Sentinel type returned when a cache lookup finds no stored value."""
    ...


class ParamCache:
    """Nested dict cache for script parameter values across Streamlit reruns.

    Cache structure::

        {script_key: {cache_type: {param_name: value}}}

    where ``cache_type`` is one of ``'option'`` (widget options list),
    ``'value'`` (current widget value), or ``'valid'`` (validation result).
    """

    def __init__(self) -> None:
        """Initialise an empty cache."""
        self.cache : dict[str, dict[str, dict[str, Any]]] = {}

    def __repr__(self) -> str:
        """Return a debug string showing the full cache contents."""
        return f"ParamCache(cache={self.cache})"

    def __bool__(self) -> bool:
        """Return True."""
        return True

    def has(self, key: str , cache_type : Literal['option', 'value', 'valid'] , name : str) -> bool:
        """Return True if a value is stored for the given ``(script_key, cache_type, name)``."""
        return name in self.cache.get(key, {}).get(cache_type, {})

    def get(self, key: str , cache_type : Literal['option', 'value', 'valid'] , name : str) -> Any:
        """Retrieve a cached value; raises ``KeyError`` if not present (use :meth:`has` first)."""
        return self.cache.get(key, {}).get(cache_type, {})[name]

    def set(self, value : Any, key: str, cache_type : Literal['option', 'value', 'valid'] , name : str) -> None:
        """Store *value* under the given ``(script_key, cache_type, name)`` triple."""
        if key not in self.cache:
            self.cache[key] = {}
        if cache_type not in self.cache[key]:
            self.cache[key][cache_type] = {}
        self.cache[key][cache_type][name] = value

    def init_cache(self, key: str) -> None:
        """Ensure all three cache-type sub-dicts exist for *key*."""
        if key not in self.cache:
            self.cache[key] = {}
        for cache_type in ['option', 'value', 'valid']:
            if cache_type not in self.cache[key]:
                self.cache[key][cache_type] = {}

    def clear_cache(self, key: str) -> None:
        """Clear all cached values for *script_key* and re-initialise the sub-dicts."""
        if key in self.cache:
            self.cache[key].clear()
        self.init_cache(key)

    def update_cache(self, key: str, cache_type: Literal['option', 'value', 'valid'], dict_values: dict[str, Any]) -> None:
        """Bulk-set multiple values for the given *script_key* and *cache_type*."""
        self.init_cache(key)
        for name, value in dict_values.items():
            self.set(value, key, cache_type, name)

@dataclass
class WidgetParamInput:
    """Streamlit-aware extension of :class:`ScriptParamInput`."""
    runner_key: str
    name: str
    type: Literal['str', 'int', 'float', 'bool', 'list', 'tuple' , 'enum'] | list[str] | tuple[str]
    desc: str
    required: bool = False
    default: Any = None
    min: Any = None
    max: Any = None
    prefix: str = ''
    enum: list[str] | None = None
    disabled: bool = False

    def __post_init__(self) -> None:
        """Initialise by copying *param* fields and building the option ↔ value transformers."""
        self.option_to_value = self.option_to_value_transformer()
        self.value_to_option = self.value_to_option_transformer()

    @classmethod
    def from_script_param_input(cls , runner_key : str , param : ScriptParamInput) -> WidgetParamInput:
        """Create a :class:`WidgetParamInput` from a :class:`ScriptParamInput`."""
        return cls(runner_key, **param.as_dict())

    @classmethod
    def from_endpoint_parameter(cls , runner_key : str , param : dict[str, Any]) -> WidgetParamInput:
        """Create a :class:`WidgetParamInput` from a :class:`APIEndpoint` parameter."""
        return cls(runner_key = runner_key, **signature_row_to_input_param(param))

    @property
    def ptype(self) -> type | list[str]:
        """Resolve ``type`` to a Python type object or list of valid string options."""
        if isinstance(self.type, str):
            if self.type == 'str':
                ptype = str
            elif self.type == 'int':
                ptype = int
            elif self.type == 'float':
                ptype = float
            elif self.type == 'bool':
                ptype = bool
            elif self.type in ['list', 'tuple' , 'enum']:
                assert self.enum , f'enum is required for {self.type}'
                ptype = list(self.enum)
            else:
                try:
                    ptype = eval(self.type)
                except Exception as e:
                    Logger.warning(e)
                    Logger.warning(f'Invalid type: {self.type} , using str as default')
                    ptype = str
        elif isinstance(self.type, (list, tuple)):
            ptype = list(self.type)
        else:
            raise ValueError(f'Invalid type: {self.type}')
        return ptype

    @property
    def title(self) -> str:
        """Human-readable title derived from the parameter name (snake_case → Title Case)."""
        title = self.name.replace('_', ' ').title()
        return title

    @property
    def placeholder(self) -> str:
        """Placeholder text for text/number widgets; falls back to ``name`` when ``desc`` is empty."""
        placeholder = self.desc if self.desc else self.name
        return placeholder

    def is_valid(self) -> bool:
        """Return True if *value* satisfies the ``required`` constraint."""
        if self.required:
            return self.param_value not in ['', None , 'Choose an option']
        return True

    def error_message(self) -> str | None:
        """Return a user-facing error string if *value* is invalid, otherwise None."""
        if not self.is_valid():
            operator = 'input' if self.type in ['str', 'int', 'float'] else 'select'
            return f"Please {operator} a valid value for [{self.title}]"
        return None

    @property
    def option(self) -> Any:
        """Current raw widget option from ``st.session_state`` (may be None)."""
        return st.session_state.get(self.widget_key, None)

    @property
    def param_value(self) -> Any:
        """The typed Python value derived from the current widget option."""
        return self.option_to_value(self.option)

    @property
    def widget_key(self) -> str:
        """Unique Streamlit session-state key for this widget."""
        return f"script-param-{self.runner_key}-{self.name}"
    
    def option_to_value_transformer(self) -> Callable:
        """Build a callable that converts a raw widget option to its typed Python value."""
        ptype = self.ptype
        if isinstance(ptype, list):
            options = ['Choose an option'] + [f'{self.prefix}{e}' for e in ptype]
            values = [None] + ptype
            def wrapper(option : Any):
                """get index of value in options"""
                if option is None or option == '' or option == 'Choose an option': 
                    option = 'Choose an option'
                if option not in options:
                    option = self.remove_extra_prefix_regex(option, self.prefix, remain_prefix = False)
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

    def value_to_option_transformer(self) -> Callable:
        """Build a callable that converts a typed Python value back to its widget option string."""
        ptype = self.ptype
        if isinstance(ptype, list):
            options = ['Choose an option'] + [f'{self.prefix}{e}' for e in ptype]
            values = [None] + [str(ptype_e) for ptype_e in ptype]
            def wrapper(value : Any):
                """get index of value in options"""
                if value is None or value == '' or value == 'Choose an option': 
                    value = None
                if value not in values:
                    value = self.remove_extra_prefix_regex(value, self.prefix , remain_prefix = False)
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
    def remove_extra_prefix_regex(cls , s : str | Any, prefix : str , remain_prefix : bool = True) -> str:
        """Strip repeated occurrences of *prefix* from *s*, optionally re-adding it once."""
        s = str(s)
        if not prefix:
            return s
        while s.startswith(prefix):
            s = s.removeprefix(prefix)
        if remain_prefix:
            return f'{prefix}{s}'
        else:    
            return s
    
    def on_change(self , cache : ParamCache) -> None:
        """Persist current option, value, and validity into *cache* when the widget changes."""
        cache.set(self.option, self.runner_key, 'option', self.name)
        cache.set(self.param_value, self.runner_key, 'value', self.name)
        cache.set(self.is_valid(), self.runner_key, 'valid', self.name)

    def default_option(self , cache : ParamCache, value : Any | None = None) -> Any:
        """Resolve the default widget option from an explicit *value*, the cache, or the param default."""
        if value is not None:
            default_option = self.value_to_option(value)
        else:
            default_option = None
            if cache.has(self.runner_key, 'option', self.name):
                default_option = cache.get(self.runner_key, 'option', self.name)
            elif cache.has(self.runner_key, 'value', self.name):
                default_option = self.value_to_option(cache.get(self.runner_key, 'value', self.name))
            #elif self.widget_key in st.session_state:
            #    default_option = st.session_state.get(self.widget_key, None)
            else:
                default_option = self.value_to_option(self.default)
        return default_option

class ParamInputsForm:
    """Complete parameter form for a script, managing widget creation and validation.

    Wraps all :class:`WidgetParamInput` objects for a runner, handles form
    layout (``customized`` or ``form``), parses previous command strings to
    pre-fill values, and validates before submission.

    Parameters
    ----------
    runner:
        The script whose parameters this form represents.
    cache:
        Session-scoped :class:`ParamCache` used to persist widget state.
    item:
        Optional previous :class:`~src.interactive.backend.TaskItem` used to
        pre-fill values from its command string.
    """
    def __init__(self , runner_key : str , widgets : list[WidgetParamInput] , cache : ParamCache | None = None, item : TaskItem | None = None) -> None:
        assert all(w.runner_key == runner_key for w in widgets) , f"All widgets must have the same runner_key : {runner_key} , but got {[w.runner_key for w in widgets]}"
        self.runner_key = runner_key
        self.param_dict = {p.name:p for p in widgets}
        self.errors = []
        self.trigger_item = item 
        self.cache = cache or ParamCache()

    @classmethod
    def from_runner(cls , runner : ScriptRunner , cache : ParamCache | None = None, item : TaskItem | None = None) -> 'ParamInputsForm':
        """Create a :class:`ParamInputsForm` from a :class:`ScriptRunner`."""
        widgets = [WidgetParamInput.from_script_param_input(runner.script_key, p) for p in runner.header.get_param_inputs()]
        return cls(runner.script_key, widgets, cache, item)

    @classmethod
    def from_api_endpoint(cls , endpoint , cache : ParamCache | None = None, item : TaskItem | None = None) -> 'ParamInputsForm':
        """Create a :class:`ParamInputsForm` from a :class:`APIEndpoint`."""
        from src.interactive.main.util.api_adapter import stAPIEndpoint
        assert isinstance(endpoint, stAPIEndpoint) , f"Endpoint {endpoint} is not a stAPIEndpoint"
        frozen_widget = WidgetParamInput(
            runner_key = endpoint.runner_key, 
            name = 'qualname', type = 'str' , desc = 'Endpoint Qualname' , default = endpoint.qualname , disabled = True)
        widgets = [frozen_widget] + [WidgetParamInput.from_endpoint_parameter(endpoint.runner_key, p) for p in endpoint.parameters]
        return cls(endpoint.runner_key, widgets, cache, item)

    def init_param_inputs(self , type : Literal['customized', 'form'] = 'customized' , cmd : str | None = None) -> ParamInputsForm:
        """Render the parameter form widgets.

        Parameters
        ----------
        type:
            ``'customized'`` renders widgets in a multi-column layout;
            ``'form'`` uses a ``st.form`` submit button.
        cmd:
            Optional previous command string; values are parsed and used to
            pre-fill widgets.  Defaults to the trigger item's ``cmd`` if set.
        """
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

    def cmd_to_param_values(self , cmd : str = '') -> dict[str, Any]:
        """Parse a command string and extract ``{param_name: value}`` pairs for this script."""
        param_values = {}
        if not cmd or str(self.runner_key) not in cmd: 
            return param_values
        main_str = [s for s in cmd.split(";") if self.runner_key in s][0]
        param_str = ''.join(main_str.split(self.runner_key)[1:]).strip()
        
        for pstr in param_str.split('--'):
            if not pstr: 
                continue
            try:
                param_name , param_value = pstr.replace('=' , ' ').split(' ' , 1)
            except Exception as e:
                Logger.stdout(cmd)
                Logger.stdout(pstr)
                Logger.error(f"Error parsing param: {pstr} - {e}")
                raise

            value = param_value.strip()
            if value == 'True': 
                value = True
            elif value == 'False': 
                value = False
            elif value == 'None': 
                value = None
            param_values[param_name] = value
        return param_values

    def init_customized_container(self , cmd : str = '' , num_cols : int = 4) -> ParamInputsForm:
        """Render widgets in a responsive multi-column grid layout. Returns self."""
        num_cols = min(num_cols, len(self.param_dict))
        cmd_param_values = self.cmd_to_param_values(cmd)
        for i, wp in enumerate[WidgetParamInput](self.param_dict.values()):
            if i % num_cols == 0:
                param_cols = st.columns(num_cols)
            with param_cols[i % num_cols]:
                self.get_widget(
                    param=wp, 
                    cache=self.cache,
                    value = cmd_param_values.get(wp.name) ,
                    on_change=self.on_widget_change, args=(wp, self.cache))
                if err_msg := wp.error_message():
                    st.error(err_msg , icon = ":material/error:")
        return self
    
    def init_form(self , cmd : str) -> ParamInputsForm:
        """Render all widgets inside a ``st.form`` container with a Submit button. Returns self."""
        cmd_param_values = self.cmd_to_param_values(cmd)
        with st.form(f"ParamInputsForm-{self.runner_key}" , clear_on_submit = False):
            for param in self.param_dict.values():
                self.get_widget(param, cache=self.cache, value = cmd_param_values.get(param.name))

            if st.form_submit_button(
                "Submit" ,
                help = "Submit Parameters to Run Script" ,
            ):
                self.submit()

        return self

    @property
    def param_values(self) -> dict[str, Any]:
        """Return the current ``{param_name: typed_value}`` dict for all parameters."""
        return {wp.name: wp.param_value for wp in self.param_dict.values()}

    def validate(self) -> bool:
        """Validate all widgets; populate ``self.errors`` and return True if all pass."""
        self.errors = []
        for wp in self.param_dict.values():
            if err_msg := wp.error_message():
                self.errors.append(err_msg)

        return len(self.errors) == 0

    def submit(self) -> None:
        """Persist all widget values to cache and display any validation errors."""
        for wp in self.param_dict.values():
            wp.on_change(self.cache)

        if not self.validate():
            for err_msg in self.errors:
                st.error(err_msg , icon = ":material/error:")

    def process(self) -> None:
        """Placeholder for post-submit processing logic (not yet implemented)."""
        ...

    def reset_options(self) -> None:
        """Reset the options of the parameter inputs form into the cache values."""
        for wp in self.param_dict.values():
            if wp.widget_key in st.session_state and st.session_state.get(wp.widget_key) != wp.default_option(self.cache):
                del st.session_state[wp.widget_key]
                st.session_state[wp.widget_key] = wp.default_option(self.cache)
        st.rerun()

    @classmethod
    def get_widget(cls , param : WidgetParamInput , cache : ParamCache, value : Any | None = None ,
                   on_change : Callable | None = None , args : tuple | None = None , kwargs : dict | None = None) -> Any:
        """Dispatch to the correct widget factory based on ``param.ptype``.

        Parameters
        ----------
        param:
            The parameter descriptor.
        cache:
            Session-scoped parameter cache.
        value:
            Optional explicit value to pre-fill (overrides cache).
        on_change, args, kwargs:
            Forwarded to the underlying Streamlit widget callback.
        """
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
        
        return func(param, cache, value, on_change = on_change, args = args, kwargs = kwargs)
    
    @classmethod
    def get_title(cls , param : WidgetParamInput) -> str:
        """Return a Streamlit markdown label; required params are styled in red with an asterisk."""
        return f':red[:material/asterisk: **{param.title}**]' if param.required else f'**{param.title}**'

    @classmethod
    def get_help(cls , param : WidgetParamInput) -> str:
        """Build the hover-help string, combining required notice and description."""
        help_texts = []
        if param.required: 
            help_texts.append(f':red[**Required Parameter!**]')
        if param.desc: 
            help_texts.append(f'*{param.desc}*')
        return '\t'.join(help_texts)

    @classmethod
    def list_widget(cls , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs) -> Any:
        """Render a ``st.selectbox`` (or ``st.text_input`` for unknown options) for list/enum params."""
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
                disabled=param.disabled,
                **kwargs
            )
    
    @classmethod
    def text_widget(cls , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs) -> Any:
        """Render a ``st.text_input`` for string params."""
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
            disabled=param.disabled,
            **kwargs
        )
    
    @classmethod
    def bool_widget(cls , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs) -> Any:
        """Render a ``st.selectbox`` with True / False / 'Choose an option' for bool params."""
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
            disabled=param.disabled,
            **kwargs
        )
    
    @classmethod
    def int_widget(cls , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs) -> Any:
        """Render a ``st.number_input`` (integer) for int params."""
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
            disabled=param.disabled,
            help=help,
            **kwargs
        )
    
    @classmethod
    def float_widget(cls , param : WidgetParamInput , cache : ParamCache, value : Any | None = None , **kwargs) -> Any:
        """Render a ``st.number_input`` (float, step 0.1) for float params."""
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
            disabled=param.disabled,
            **kwargs
        )

    @classmethod
    def get_form_errors(cls) -> Any:
        """Return the current form errors from ``st.session_state``."""
        return st.session_state.form_errors

    @classmethod
    def on_widget_change(cls , wp : WidgetParamInput , cache : ParamCache) -> None:
        """Callback wired to widget ``on_change``; delegates to :meth:`WidgetParamInput.on_change`."""
        wp.on_change(cache)

def _matching_bracket(s: str, i: int) -> int | None:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "[":
            depth += 1
        elif s[j] == "]":
            depth -= 1
            if depth == 0:
                return j
    return None


def _literal_inners(ann: str) -> list[str]:
    parts: list[str] = []
    pos = 0
    while pos < len(ann):
        m = re.compile(r"(?:typing\.)?Literal\s*\[").search(ann, pos)
        if not m:
            break
        lb = ann.find("[", m.start())
        rb = _matching_bracket(ann, lb)
        if rb is None:
            break
        parts.append(ann[lb + 1 : rb])
        pos = rb + 1
    return parts


def _parse_literal_inner(inner: str) -> list[str]:
    inner = inner.strip()
    if not inner:
        return []
    try:
        tup = ast.literal_eval("(" + inner + ")")
    except (SyntaxError, ValueError):
        return []
    if not isinstance(tup, tuple):
        tup = (tup,)
    return [v if isinstance(v, str) else str(v) for v in tup]


def _dedupe_preserve(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _all_literal_strings(ann: str) -> list[str]:
    acc: list[str] = []
    for seg in _literal_inners(ann):
        acc.extend(_parse_literal_inner(seg))
    return _dedupe_preserve(acc)


def _unwrap_optional(ann: str) -> str | None:
    a = ann.strip()
    m = re.match(r"^(typing\.)?Optional\s*\[", a)
    if not m:
        return None
    lb = a.index("[", m.start())
    rb = _matching_bracket(a, lb)
    if rb is None or rb != len(a) - 1:
        return None
    return a[lb + 1 : rb].strip()


def _strip_optional_and_none_union(ann: str) -> str:
    a = ann.strip()
    for _ in range(16):
        inner = _unwrap_optional(a)
        if inner is None:
            break
        a = inner
    parts = [p.strip() for p in re.split(r"\s*\|\s*", a)]
    parts = [p for p in parts if p and p not in ("None", "typing.None", "types.NoneType")]
    return parts[0].strip() if len(parts) == 1 else " | ".join(parts)


def _has_collection(ann: str) -> bool:
    lo = ann.lower()
    return bool(
        re.search(r"\b(list|tuple|dict)\s*\[", lo)
        or re.search(r"\btyping\.(list|tuple|dict)\s*\[", lo)
    )

def annotation_to_type_enum(
    annotation: str,
    *,
    default: Any = None,
) -> tuple[Literal["str", "int", "float", "bool", "enum"], list[str] | None]:
    """Infer ``(widget_type, enum_options)`` from a string annotation."""
    raw = (annotation or "").strip()
    if not raw:
        return "str", None
    if _has_collection(raw):
        return "str", None

    core = _strip_optional_and_none_union(raw)
    compact = re.sub(r"\s+", " ", core.strip())

    lits = _all_literal_strings(raw)
    if lits:
        return "enum", lits

    if "|" not in compact:
        key = compact.casefold()
        if key in ("bool", "int", "float", "str"):
            return key, None  # type: ignore[return-value]

    if re.search(r"\bbool\b", compact, re.I):
        return "bool", None
    if re.search(r"\bint\b", compact, re.I) and "point" not in compact.lower():
        return "int", None
    if re.search(r"\bfloat\b", compact, re.I):
        return "float", None
    if re.search(r"\bstr\b", compact, re.I):
        return "str", None
    if re.search(r"\bAny\b", compact):
        if isinstance(default, bool):
            return "bool", None
        if isinstance(default, int) and not isinstance(default, bool):
            return "int", None
        if isinstance(default, float):
            return "float", None
    return "str", None

def signature_row_to_input_param(row: dict[str, Any]) -> dict[str, Any]:
    """Turn one ``{name, annotation, default, override?}`` row into a ``from_endpoint_parameter`` dict."""
    name = row["name"]
    ann = str(row.get("annotation") or "")
    default = row.get("default", None)
    ovr = row["override"] if isinstance(row.get("override"), dict) else {}
    required = 'default' not in ovr and 'default' not in row

    eff_default = ovr["default"] if "default" in ovr else default
    desc = str(ovr.get("desc") or "")

    if "type" in ovr:
        t = str(ovr["type"])
        if t not in ["str", "int", "float", "bool", "enum"]:
            raise ValueError(f"invalid override.type {t!r} for parameter {name!r}")
        wtype: Literal["str", "int", "float", "bool", "enum"] = t  # type: ignore[assignment]
        if wtype == "enum":
            opts = ovr.get("enum")
            if not isinstance(opts, list) or not opts:
                raise ValueError(f"override.type 'enum' requires non-empty override.enum for {name!r}")
            enum: list[str] | None = [str(x) for x in opts]
        else:
            enum = None
    else:
        wtype, enum = annotation_to_type_enum(ann, default=eff_default)
        if "enum" in ovr:
            eo = ovr["enum"]
            if not isinstance(eo, list) or not eo:
                raise ValueError(f"override.enum must be a non-empty list for {name!r}")
            wtype, enum = "enum", [str(x) for x in eo]

    if wtype == "enum" and not enum:
        raise ValueError(f"enum widget has no options for {name!r} (annotation={ann!r})")

    out: dict[str, Any] = {
        "name": name,
        "type": wtype,
        "desc": desc,
        'required': required,
        "enum": enum,
    }
    if eff_default is not None or "default" in row or "default" in ovr:
        out["default"] = eff_default
    return out
