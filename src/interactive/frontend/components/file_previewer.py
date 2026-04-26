"""
Streamlit frontend utilities for the interactive application.

FilePreviewer
    Multi-format file viewer supporting text, HTML, PDF, and Excel.

"""
from __future__ import annotations
from pathlib import Path
import streamlit as st
import base64
import streamlit.components.v1 as components
import pandas as pd

from src.proj.core import strPath

class FilePreviewer:
    """Multi-format file viewer that renders file content inside a Streamlit container.

    Supported formats: ``.txt``, ``.csv``, ``.json``, ``.log``, ``.py`` (code blocks),
    ``.html`` (iframe), ``.pdf`` (base64 iframe), ``.xlsx`` / ``.xls`` (dataframe).
    """
    def __init__(self , path : strPath | None = None , height : int | None = 600) -> None:
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