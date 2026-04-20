"""
Streamlit frontend utilities for the interactive application.

Classes
-------
ActionLogger
    Singleton that records user interactions and errors to log files.
FilePreviewer
    Multi-format file viewer supporting text, HTML, PDF, and Excel.
YAMLFileEditorState / YAMLFileEditor
    Interactive YAML file editor with load/validate/save and a streamlit-ace widget.
ColoredText
    String subclass that auto-applies Streamlit badge colours based on log level.

Module-level helpers
--------------------
expander_subheader
    Render a custom expandable section header with optional help tooltip.
action_confirmation
    ``st.dialog``-based confirmation dialogue for destructive actions.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal , Any , Callable , Sequence
import streamlit as st
import base64 , yaml , shutil
import streamlit.components.v1 as components
import pandas as pd

from datetime import datetime
from dataclasses import dataclass

from src.proj import PATH , Logger

class ActionLogger:
    """Singleton logger that records UI action events and errors to flat log files.

    Log files are stored in ``PATH.app_db``.  Use :meth:`log_action` as a
    decorator on action handlers to capture timing and arguments automatically.
    """
    @staticmethod
    def get_log_path(log_type : Literal['action' , 'error']) -> Path:
        """Return the path to the action or error log file, creating it if absent."""
        file_path = PATH.app_db / f'page_{log_type}.log'
        file_path.touch(exist_ok=True)
        return file_path
    
    action_log = get_log_path('action')
    error_log  = get_log_path('error')
    _instance = None
    _ignores = None

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register_ignore(cls , ignore : str | list[str] | None = None) -> None:
        """Register argument names/types to exclude from all subsequent log entries."""
        if ignore is None:
            ignore = []
        elif isinstance(ignore , str):
            ignore = [ignore]
        assert isinstance(ignore , list) , f'ignore must be None or a string or a list of strings'
        if cls._ignores is None:
            cls._ignores = []
        cls._ignores.extend(ignore)
    
    @classmethod
    def log_action(cls , ignore : str | list[str] | None = None) -> Callable:
        """Decorator that wraps a function to log its invocation and any exceptions.

        Parameters
        ----------
        ignore:
            Argument names or class names to omit from the logged output.
        """
        if isinstance(ignore , str):
            ignore = [ignore]
        assert not ignore or isinstance(ignore , list) , f'ignore must be None or a string or a list of strings'
            
        def wrapper(func : Callable):
            def inner(*args , **kwargs):
                t0 = datetime.now()
                try:
                    func(*args , **kwargs)
                    cls.record_action(t0 , func.__name__ , *args , **kwargs)
                except Exception as e:
                    err_msg = Logger.print_exc(e)
                    cls.record_error(t0 , func.__name__ , err_msg)
                    raise
            return inner
        return wrapper

    @classmethod
    def record_action(cls , start_time : datetime , action : str , *args , ignore : list[str] | None = None , **kwargs) -> None:
        """Append a structured action record to the action log file."""
        args_str , kwargs_str = cls.get_args_str(args , kwargs , ignore)
        export_str = '\n'.join([
            f'Action: [{action}]', 
            f'    Start:  [{start_time.strftime("%Y-%m-%d %H:%M:%S")}]', 
            f'    End:    [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', 
            f'    Args:   [{args_str}]', 
            f'    Kwargs: [{kwargs_str}]'])
        with open(cls.action_log, 'a') as f:
            f.write(export_str + '\n')

    @classmethod
    def get_args_str(cls , args : list[Any] | tuple[Any, ...] , kwargs : dict[str, Any] , ignore : list[str] | None = None) -> tuple[str, str]:
        """Format positional and keyword arguments as loggable strings, applying ignores."""
        if ignore is None: 
            ignore = []
        if cls._ignores: 
            ignore.extend(cls._ignores)
        def get_name(x : Any):
            return f"{x.__class__.__name__}({x.id})" if hasattr(x, 'id') else str(x)
        args_str = ', '.join([get_name(arg) for arg in args if not cls.ignore_arg(arg , ignore)])
        kwargs_str = ', '.join([f"{k}: {(get_name(v))}" for k, v in kwargs.items() if not cls.ignore_arg(v , ignore , k)])
        return args_str , kwargs_str
    
    @classmethod
    def ignore_arg(cls , obj : Any , ignore : list[str] | None = None , key : str | None = None) -> bool:
        """Return True if *obj* or *key* matches any entry in the *ignore* list."""
        if ignore is None: 
            ignore = []
        arg_in = (
            str(obj) in ignore or
            (hasattr(obj , '__name__') and obj.__name__ in ignore) or
            obj.__class__.__name__ in ignore)
        key_in = key is not None and key in ignore
        return arg_in or key_in

    @classmethod
    def record_error(cls , start_time : datetime , action : str , error : str) -> None:
        """Append a structured error record to the error log file."""
        export_str = '\n'.join([
            f'Action: [{action}]', 
            f'    Start:  [{start_time.strftime("%Y-%m-%d %H:%M:%S")}]', 
            f'    End:    [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', 
            f'    Error:  [{error}]'])
        with open(cls.error_log, 'a') as f:
            f.write(export_str + '\n')

    @classmethod
    def clear_log(cls , suffix : str | None = None) -> None:
        """Rotate and archive both log files, then recreate empty ones.

        Parameters
        ----------
        suffix:
            Timestamp suffix used for the backup filenames; defaults to now.
        """
        if suffix is None: 
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        for log in [cls.action_log , cls.error_log]:
            if log.exists():
                backup_path = log.parent.joinpath('backup', f'{log.name}_{suffix}.log')
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(log , backup_path)
                log.unlink(missing_ok=True)
            log.touch()

    @classmethod
    def get_action_log(cls) -> str:
        """Return the full text content of the action log, creating the file if absent."""
        if not cls.action_log.exists():
            cls.action_log.touch()
        return cls.action_log.read_text()

    @classmethod
    def get_error_log(cls) -> str:
        """Return the full text content of the error log, creating the file if absent."""
        if not cls.error_log.exists():
            cls.error_log.touch()
        return cls.error_log.read_text()
    
class FilePreviewer:
    """Multi-format file viewer that renders file content inside a Streamlit container.

    Supported formats: ``.txt``, ``.csv``, ``.json``, ``.log``, ``.py`` (code blocks),
    ``.html`` (iframe), ``.pdf`` (base64 iframe), ``.xlsx`` / ``.xls`` (dataframe).
    """
    def __init__(self , path : Path | str | None = None , height : int | None = 600) -> None:
        self.path = Path(path) if path is not None else None
        self.height = height or 600

    def preview(self) -> None:
        """Render the file contents into the current Streamlit page."""
        if self.path is None: 
            return
        elif not self.path.exists():
            st.error(f"File {self.path} not found" , icon = ":material/error:")
            return
        st.markdown(f"**File Path:** {self.path.absolute()}")
        with st.container(height = self.height , border = False):
            # st.info(f"Previewing file: {self.path}" , icon = ":material/file_present:" , width = "stretch")
            suffix = self.path.suffix
            if suffix in ['.txt', '.csv', '.json' , '.log' , '.py']:
                language = {
                    '.txt': None,
                    '.csv': 'csv',
                    '.json': 'json',
                    '.log': 'log',
                    '.py': 'python',
                }[suffix]
                self.preview_text_file(self.path , language)
            elif suffix == '.html':
                self.preview_html_file(self.path)
            elif suffix == '.pdf':
                self.preview_pdf_file(self.path)
            elif suffix in ['.xlsx' , '.xls']:
                self.preview_xlsx_file(self.path)
            else:
                self.preview_not_supported(self.path)

    @staticmethod
    def preview_text_file(file_path : Path , language : str | None = None):
        """preview text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            if language == 'json':
                st.json(text_content)
            else:
                st.code(text_content , language=language , width = "stretch")
        except Exception as e:
            st.error(f"Cannot preview text file: {str(e)}" , icon = ":material/error:")

    @staticmethod
    def preview_html_file(file_path : Path):
        """preview HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content , height = 600 , scrolling=True)   
        except Exception as e:
            st.error(f"Cannot preview HTML file: {str(e)}" , icon = ":material/error:")

    @staticmethod
    def preview_pdf_file(file_path):
        """preview PDF file"""
        try:
            with open(file_path, 'rb') as f:
                pdf_data = f.read()
            
            # use base64 to encode PDF
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            pdf_display = f'''
            <iframe src="data:application/pdf;base64,{pdf_base64}" 
                    width="200%" height="600px" type="application/pdf">
            </iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Cannot preview PDF file: {str(e)}" , icon = ":material/error:")

    @staticmethod
    def preview_xlsx_file(file_path):
        """preview XLSX file"""
        try:
            sheet_names = pd.ExcelFile(file_path).sheet_names
            sheet_name = st.selectbox("**Select sheet** :material/table_rows:", sheet_names)
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Cannot preview XLSX file: {str(e)}" , icon = ":material/error:")

    @staticmethod
    def preview_not_supported(file_path : Path):
        """preview not supported file"""
        st.error(f"Not supported to preview {file_path.suffix} file: {file_path}" , icon = ":material/error:")

@dataclass
class YAMLFileEditorState:
    """Persistent state for a :class:`YAMLFileEditor` instance, stored in ``st.session_state``.

    Attributes
    ----------
    key:
        Unique identifier for this editor instance.
    root:
        Base directory for resolving relative file paths.
    path:
        Current relative file path (as a string).
    path_status, content_status, load_status, save_status:
        Status strings; ``'success'`` means OK, anything else is an error message.
    """
    key  : str
    root : Path = Path('')
    path : str = ""
    path_status : str = 'success'
    content_status : str = 'success'
    load_status : str = 'success'
    save_status : str = 'success'

    def __post_init__(self) -> None:
        """Post-init hook (currently a no-op placeholder)."""
        ...

    @classmethod
    def get_state(cls , key : str , root : Path | str | None = None) -> 'YAMLFileEditorState':
        """Retrieve or create a :class:`YAMLFileEditorState` from ``st.session_state``.

        Parameters
        ----------
        key:
            Unique editor key.
        root:
            If provided, update the state's root directory.
        """
        if f'yaml_file_editor_states' not in st.session_state:
            st.session_state.yaml_file_editor_states = {}
        if key not in st.session_state.yaml_file_editor_states:
            st.session_state.yaml_file_editor_states[key] = cls(key)
        if root is not None:
            st.session_state.yaml_file_editor_states[key].root = Path(root)
        return st.session_state.yaml_file_editor_states[key]

    @property
    def editor_key(self) -> str:
        """Unique Streamlit session-state key for this editor's ace widget."""
        return f"file-editor-{self.key}-{self.path}"

    @property
    def load_content(self) -> str:
        """The last content loaded from disk (stored in session state)."""
        return getattr(st.session_state , f'{self.editor_key}-load' , '')

    @load_content.setter
    def load_content(self , value : str) -> None:
        """Persist newly loaded content into session state."""
        st.session_state[f'{self.editor_key}-load'] = value

    @property
    def edit_content(self) -> str:
        """The live content currently in the ace editor widget (from session state)."""
        return getattr(st.session_state , f'{self.editor_key}'  , '')

class YAMLFileEditor:
    """Interactive YAML file editor with load / validate / save controls.

    Implemented as a per-key singleton so the same editor instance is reused
    across Streamlit reruns.  Uses ``streamlit_ace`` for the editor widget.

    Parameters
    ----------
    key:
        Unique identifier for this editor instance.
    file_root:
        Root directory (or exact file path when ``file_input=False``).
    file_input:
        If True, show a path input/selectbox widget so users can choose the file.
    height:
        Pixel height of the ace editor widget.
    """
    _instances : dict = {}
    def __new__(cls , key : str = 'yaml_file_editor' , *args , **kwargs):
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]
    
    def __init__(self , key : str = 'yaml_file_editor' , file_root : Path | str = PATH.main , 
                 file_input = True , height : int | None = 500):
        self.key = key
        self.file_root = Path(file_root)
        self.file_input = file_input
        self.height = height or 500
        if not file_input:
            # assert self.file_root.is_file() , f"File root is not a file: {self.file_root}"
            pass
        else:
            assert self.file_root.exists() , f"File root does not exist: {self.file_root}"
            assert self.file_root.is_dir() , f"File root is not a directory: {self.file_root}"
        self.init_session_state()

    def __repr__(self) -> str:
        """Return a debug string showing the key and root directory."""
        return f"YAMLFileEditor(key={self.key},root={self.file_root})"

    def init_session_state(self) -> 'YAMLFileEditor':
        """Sync the editor state from ``st.session_state`` and return self."""
        self.state = YAMLFileEditorState.get_state(key = self.key , root = self.file_root)
        return self

    def get_file_root(self) -> Path:
        """Return the current root directory from the editor state."""
        return self.state.root

    def set_file_path(self , file_path : Path | str) -> None:
        """Update the relative file path in the editor state."""
        self.state.path = str(file_path)

    def get_file_path(self , file_path : Path | str | None = None) -> Path:
        """Resolve the absolute target path from root + relative path (or explicit override)."""
        root = self.get_file_root()
        if root.is_file():
            file_path = root
        elif file_path is None:
            file_path = root.joinpath(self.state.path)
        else:
            file_path = root.joinpath(file_path)
        return file_path
    
    def validate_file(self , file_path : Path | str | None = None) -> None:
        """Check that the resolved path exists and has a ``.yaml`` extension; update ``state.path_status``."""
        file_path = self.get_file_path(file_path)
        if file_path.suffix != '.yaml':
            self.state.path_status = f'File is not YAML: {file_path}'
        elif not file_path.exists():
            self.state.path_status = f'File does not exist: {file_path}'
        else:
            self.state.path_status = 'success'
    
    def validate_file_content(self , file_content : str | None = None) -> None:
        """Parse the YAML content and update ``state.content_status`` accordingly."""
        if file_content is None:
            file_content = self.state.edit_content
        if file_content is None:
            self.state.content_status = 'File content is None'
        else:
            try:
                yaml.safe_load(file_content)
                self.state.content_status = 'success'
            except yaml.YAMLError as e:
                self.state.content_status = f'YAML syntax error: {e}'

    def load_file(self , file_path : Path | str | None = None) -> str:
        """Load the target file into ``state.load_content`` and return its text (or '' on error)."""
        self.validate_file(file_path)
        if self.state.path_status != 'success':
            st.error(self.state.path_status , icon = ":material/error:")
            return ''
        file_path = self.get_file_path(file_path)
        try:
            with open(file_path, "r") as f:
                self.state.load_content = f.read() or ''
            self.state.load_status = 'success'
        except Exception as e:
            self.state.load_content = ''
            self.state.load_status = f'Load file failed: {e}'
        return self.state.load_content

    def save_file(self , file_path : Path | str | None = None) -> None:
        """Validate YAML syntax then write ``state.edit_content`` to disk."""
        file_path = self.get_file_path(file_path)
        try:
            if self.state.edit_content is None:
                raise ValueError("Edit content is None")
            yaml.safe_load(self.state.edit_content)
            with open(file_path, "w") as f:
                f.write(self.state.edit_content)
            self.state.save_status = 'success'
            st.success(f"File saved successfully : {file_path}")
        except yaml.YAMLError as e:
            self.state.save_status = f'YAML syntax error: {e}'
        except Exception as e:
            self.state.save_status = f'Save file failed: {e}'

    def init_path_input(self , file_path : Path | str | Sequence[Path | str] | None = None , default_file : Path | str | None = None) -> None:
        """Render a text input or selectbox for choosing the YAML file path."""
        if file_path is None and default_file is not None:
            file_path = default_file
        if file_path is None or isinstance(file_path , (Path , str)):
            if not self.state.path and file_path is not None: 
                self.state.path = str(file_path)
            st.text_input(f":blue-badge[:material/edit_document:**Input File Path : {self.state.root}/**]", 
                          value=self.state.path, 
                          key=f"{self.key}-file-input" ,
                          on_change = self.text_input_on_change , 
                          icon = ":material/edit_document:" ,
                          help = "Input the YAML file path")
            self.input_type = 'input'
        elif isinstance(file_path , Sequence):
            root = self.get_file_root()
            options = [Path(path).absolute().relative_to(root) for path in file_path]
            if not self.state.path and default_file is not None: 
                self.state.path = str(Path(default_file).absolute().relative_to(root))
            if not self.state.path:
                index = None
            elif Path(self.state.path) in options:
                index = options.index(Path(self.state.path))
            else:
                raise ValueError(f"Default file is not in options: {self.state.path} , options: {options}")
            st.selectbox(f":blue-badge[:material/edit_document: **Select File Path : {self.state.root}/**]", options ,
                        key=f"{self.key}-file-select" , index = index ,
                        on_change = self.selectbox_on_change , help = "Select the YAML file" )
            self.input_type = 'select'
        else:
            raise ValueError(f"Invalid file path: {file_path}")

    @st.fragment
    def show_yaml_editor(self , file_path : Path | str | Sequence[Path | str] | None = None , default_file : Path | str | None = None) -> None:
        """Render the full YAML editor widget (path input, ace editor, Reload/Validate/Save buttons).

        Decorated with ``@st.fragment`` so it re-runs independently of the page.
        """
        from streamlit_ace import st_ace
        self.init_session_state()
        if self.file_input: 
            self.init_path_input(file_path , default_file)
        if not self.load_file(): 
            return
        
        st.markdown(f"**File Path:** {self.get_file_path().absolute()}")
        st_ace(
            value = self.state.load_content,
            height = self.height ,
            language="yaml",
            theme="chrome",
            font_size=14,
            key=self.state.editor_key
        )

        cols = st.columns(3 , gap = 'small')
        with cols[0]:
            if st.button("Reload (double click)", key = f"yaml-file-editor-{self.key}-reload" ,
                         on_click = self.on_reload_file , help = "Reload the YAML file"):
                st.rerun()

        with cols[1]:
            if st.button("Validate" , key = f"yaml-file-editor-{self.key}-validate" ,
                         on_click = self.on_validate_content , help = "Validate the YAML file"):
                if self.state.path_status == 'success':
                    st.success("YAML file path is valid")
                else:
                    st.error(self.state.path_status , icon = ":material/error:")
                
                if self.state.load_status == 'success':
                    st.success("YAML file loaded successfully")
                else:
                    st.error(self.state.load_status , icon = ":material/error:")
                if self.state.content_status == 'success':
                    st.success("YAML syntax is valid")
                else:
                    st.error(self.state.content_status , icon = ":material/error:")
                
        with cols[2]:
            if self.state.save_status == 'success':
                disabled , help = False , "Save the YAML file"
            else:
                disabled , help = True , self.state.save_status
            st.button("Save", key = f"yaml-file-editor-{self.key}-save" ,
                      on_click=self.on_save_file , disabled = disabled , help = help)
        
        with st.expander("YAML preview"):
            try:
                data = yaml.safe_load(self.state.edit_content or '')
                with st.container(height = 500):
                    st.json(data)
            except Exception as e:
                st.warning(f"Cannot parse YAML: {e}")

    def path_input_on_change(self , st_widget_key : str | None = None) -> None:
        """Update the state path when the file path widget changes."""
        if st_widget_key is None:
            st_widget_key = f"{self.key}-file-{self.input_type}"
        self.set_file_path(st.session_state[st_widget_key])
        # self.load_file()

    def text_input_on_change(self) -> None:
        """Callback for the text-input path widget."""
        self.path_input_on_change(f"{self.key}-file-input")

    def selectbox_on_change(self) -> None:
        """Callback for the selectbox path widget."""
        self.path_input_on_change(f"{self.key}-file-select")

    def on_reload_file(self) -> None:
        """Button callback: reload the file from disk."""
        self.load_file()

    def on_validate_content(self) -> None:
        """Button callback: validate the current editor content."""
        self.validate_file_content()

    def on_save_file(self) -> None:
        """Button callback: save the current editor content to disk."""
        self.save_file()


class ColoredText(str):
    """String subclass that wraps text in a Streamlit colour badge based on log-level prefix.

    Supports: ``error`` → red, ``warning`` → orange, ``info`` → green,
    ``debug`` → gray, ``critical`` → violet.  Unrecognised messages are returned unchanged.
    """
    def __init__(self , text : str) -> None:
        """Initialise and auto-detect the colour for *text*."""
        self.text = text
        self.color = self.auto_color(self.text)

    def __str__(self) -> str:
        """Return the text wrapped in a Streamlit colour badge, or plain text if no colour applies."""
        if self.color is None:
            return self.text
        else:
            if self.color in ['violet' , 'red']:
                return f":{self.color}[**{self.text}**]"
            else:
                return f":{self.color}[{self.text}]"

    @staticmethod
    def auto_color(message : str) -> str | None:
        """Return the Streamlit colour name for *message*'s log-level prefix, or None."""
        if message.lower().startswith('error'):
            return 'red'
        elif message.lower().startswith('warning'):
            return 'orange'
        elif message.lower().startswith('info'):
            return 'green'
        elif message.lower().startswith('debug'):
            return 'gray'
        elif message.lower().startswith('critical'):
            return 'violet'
        else:
            return None
        
@st.dialog("Please Confirm Your Action")
def action_confirmation(on_confirm : Callable[[], None] , on_abort : Callable[[], None] | None = None ,
                        title : str = "Are You Sure about This?") -> None:
    """``st.dialog``-based confirmation dialogue with Confirm / Abort buttons.

    Parameters
    ----------
    on_confirm:
        Zero-argument callable executed when the user clicks Confirm.
    on_abort:
        Optional zero-argument callable executed when the user clicks Abort.
    title:
        Warning message shown inside the dialogue.
    """
    st.error(f":material/warning:**{title}**")
    col1 , col2 = st.columns(2 , gap = 'small')
    with col1:
        if st.button("**Confirm**" , icon = ":material/check_circle:" , type = "primary" , on_click = on_confirm):
            st.rerun()
    with col2:
        if st.button("**Abort**" , icon = ":material/cancel:" , type = "secondary" , on_click = on_abort):
            st.rerun()

def estimate_text_width(text : str, font_size : int = 24) -> int:
    """Estimate the rendered pixel width of *text* given *font_size*.

    Uses approximate character widths: ~0.7× font_size for ASCII, ~1.44× for CJK.
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese_chars
    
    char_width = font_size * 0.7
    chinese_width = font_size * 1.44
    
    estimated_width = chinese_chars * chinese_width + english_chars * char_width
    return int(estimated_width)

def expander_subheader(key : str , label : str , icon : str | None = None , expanded : bool = False ,
                       height : int | Literal['content' , 'stretch'] = 'content' , help : str | None = None , status : bool = False , color : str = 'blue') -> Any:
    """Render a custom collapsible section header with an optional help tooltip.

    Parameters
    ----------
    key:
        Unique key used to build the Streamlit container key.
    label:
        Header text displayed in the expander button.
    icon:
        Optional Material icon string prepended to *label*.
    expanded:
        Whether the section starts in the open state.
    height:
        Inner container height (px, ``'content'``, or ``'stretch'``).
    help:
        If provided, a floating help tooltip is rendered beside the label.
    status:
        If True, use ``st.status`` instead of ``st.expander``.
    color:
        Streamlit badge colour for the label.

    Returns
    -------
    Streamlit container
        The inner container that callers should use as a context manager.
    """
    container_key = f'{key.replace(" " , "-").lower()}-special-expander-' + ('status' if status else 'expander')
    with st.container():
        # if help is not None:
        #     help_icon = '<span role="img" aria-label="mode_off_on icon" translate="no" style="display: inline-block;' + \
        #         'font-family: &quot;Material Symbols Rounded&quot;;user-select: none;vertical-align: bottom;overflow-wrap: normal;">help</span>'
        #     margin_left = 24
        #     if icon is not None: 
        #         margin_left += 32
        #     margin_left += estimate_text_width(label.upper())
        #     st.markdown(f"""
        #     <div class="expander-help-container">
        #         <div class="help-tooltip">
        #             {help}
        #         </div>
        #         <span class="help-icon" style="margin-left: {margin_left}px;">
        #             {help_icon}
        #         </span>
        #     </div>
        #     """, unsafe_allow_html=True)
        
        container = st.container(key = container_key)
        full_label = label if icon is None else f'{icon} {label}'
        if status:
            exp_container = container.status(f" :{color}[{full_label}]" , expanded = expanded)
        else:
            exp_container = container.expander(f":{color}[{full_label}]" , expanded = expanded).container(height = height)
        
    return exp_container