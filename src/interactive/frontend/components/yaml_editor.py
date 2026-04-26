"""
Streamlit frontend utilities for the interactive application.

YAMLFileEditorState / YAMLFileEditor
    Interactive YAML file editor with load/validate/save and a streamlit-ace widget.

"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import streamlit as st
import yaml

from dataclasses import dataclass

from src.proj import PATH

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
    def get_state(cls , key : str , root : Path | str | None = None) -> YAMLFileEditorState:
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

    def init_session_state(self) -> YAMLFileEditor:
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