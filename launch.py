#!/usr/bin/env python3
# coding: utf-8

from src.proj.util.shell_opener import Shell

if __name__ == '__main__':
    Shell.open('uv run streamlit run src/interactive/main/launch.py --server.runOnSave=True' , 
               pause_when_done=False, new_on='workspace' , title='Streamlit Server' , as_workspace = 'Streamlit Server')