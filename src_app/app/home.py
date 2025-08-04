

import sys , pathlib , os
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)
assert os.getcwd() == path , f'current working directory is not {path} , do not know where to find src file'

import streamlit as st

from util import starter , basic_info , __version__ , __recommended_explorer__

st.set_option('client.showSidebarNavigation', False)
def page_config():
    st.set_page_config(
        page_title="Script Runner",
        page_icon=":material/rocket_launch:",
        layout='wide',
        initial_sidebar_state="expanded"
    )

def main():
    page_config()
    starter()
    basic_info()
    
if __name__ == '__main__':
    main() 