#!/usr/bin/env python3
# coding: utf-8
from src.proj import MACHINE
from src.proj.util.shell_opener import Shell

if __name__ == '__main__':
    cmd = 'uv run streamlit run src/interactive/main/launch.py --server.runOnSave=True  --server.fileWatcherType none'
    kwargs = {
        'pause_when_done': False,
        'close_when_done': True,
        'title': 'Streamlit Server',
        'as_from_workspace': 'Streamlit Server',
    }
    if not MACHINE.is_macos:
        kwargs['new_on'] = 'tab'
    Shell.open(cmd , **kwargs)
