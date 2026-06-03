#!/usr/bin/env python3
# coding: utf-8
from src.proj import MACHINE
from src.proj.util.shell_opener import Shell

if __name__ == '__main__':
    cmd = 'uv run streamlit run src/interactive/main/launch.py --server.runOnSave=True  --server.fileWatcherType none'
    if MACHINE.is_macos:
        Shell.open(cmd , pause_when_done=False, title='Streamlit Server' , as_from_workspace = 'Streamlit Server')
    else:
        Shell.open(cmd , pause_when_done=False, new_on='tab' , title='Streamlit Server')
