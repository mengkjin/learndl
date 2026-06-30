#!/usr/bin/env python3
# coding: utf-8

if __name__ == '__main__':
    import os
    from src.proj.util.shell import Shell
    cmd = 'uv run --frozen python -c "from src.api.calls.launcher import DirectCallHub; DirectCallHub.go()"'
    kwargs = {
        'done_action': 'pause',
        'title': 'Streamlit Server',
        'as_from_workspace': 'Streamlit Server',
        'new_on': 'tab',
    }
    Shell.open(cmd , cwd=os.getcwd(), **kwargs)
