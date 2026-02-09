#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 

from pathlib import Path
from src.proj import DB , PATH

def find_revscreen_files():
    paths = list(set(list(PATH.main.rglob('*revscreen*')) + list(PATH.main.rglob('*rev_screen*'))))
    paths = [x for x in paths if x.is_file() and x.suffix in ['.tar' , '.feather']]
    return paths

def find_reinforce_files():
    paths = list(PATH.main.rglob('*reinforce*'))
    paths = [x for x in paths if x.is_file() and x.suffix in ['.tar' , '.feather']]
    return paths

def replace_revscreen_to_reinforce(path : Path):
    path_new = Path(str(path).replace('revscreen' , 'reinforce').replace('rev_screen' , 'reinforce').replace('Revscreen' , 'Reinforce'))
    if path.suffix == '.tar':
        dfs = DB.load_dfs_from_tar(path)
        if 'index' in dfs:
            dfs['index'] = dfs['index'].replace('revscreen' , 'reinforce').replace('rev_screen' , 'reinforce').replace('Revscreen' , 'Reinforce')
        DB.save_dfs_to_tar(dfs , path_new)
    elif path.suffix == '.feather':
        df = DB.load_df(path)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('revscreen' , 'reinforce').replace('rev_screen' , 'reinforce').replace('Revscreen' , 'Reinforce')
        DB.save_df(df , path_new)
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}')

@ScriptTool('temporary_fix')
def main(**kwargs):
    for path in find_revscreen_files():
        replace_revscreen_to_reinforce(path)
        
if __name__ == '__main__':
    main()
        
    