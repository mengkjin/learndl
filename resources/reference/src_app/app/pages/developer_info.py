import streamlit as st

from src_app.frontend.frontend import ActionLogger , expander_subheader

from util import SC , set_current_page , show_sidebar_buttons , print_page_header

PAGE_NAME = 'developer_info'

def developer_info_selected_change():
    selected = getattr(st.session_state , 'developer-info-selected' , [])
    if 'All' in selected:
        st.session_state['developer-info-selected'] = ['Session Control' , 'Session States' , 'Task Queue' , 'Action Logs' , 'Error Logs']
    if 'None' in selected:
        st.session_state['developer-info-selected'] = []
    

def show_developer_info(H = 500):
    """show developer info"""
    segments = {
        "Session Control" : {
            'icon' : ':material/settings:' ,
            'operation' : lambda : st.write(SC) 
        } , 
        "Session States"  : {
            'icon' : ':material/star:' ,
            'operation' : lambda : st.write(st.session_state) 
        } , 
        "Task Queue" : {
            'icon' : ':material/directory_sync:' ,
            'operation' : lambda : st.json(SC.task_queue.queue_content() , expanded = 1)
        } , 
        "Action Logs" : {
            'icon' : ':material/format_list_numbered:' ,
            'operation' : lambda : st.code(ActionLogger.get_action_log(), language='log' , height = H , wrap_lines = True)
        } , 
        "Error Logs" : {
            'icon' : ':material/error:' ,
            'operation' : lambda : st.code(ActionLogger.get_error_log(), language='log' , height = H , wrap_lines = True)
        }}
    
    with st.container(key = "developer-info-special-expander"):
        col_name , col_widget = st.columns([1,3])
        col_name.info('Select Developer Info Types')
        col_widget.segmented_control('developer-info-selected' , 
                                options = ["All" , "None"] + list(segments.keys()) , 
                                key = "developer-info-selected" , 
                                selection_mode = "multi" ,
                                default = list(segments.keys()) , label_visibility = "collapsed" ,
                                on_change = developer_info_selected_change)
        
        col_name , col_widget = st.columns([1,3])
        col_name.info('Developer Level Operations')
        cols = col_widget.columns(7)
        cols[0].button("Log" , icon = ":material/delete_forever:" , key = "developer-log-clear" , 
                      help = "Clear Both Action and Error Logs" ,
                      on_click = SC.click_log_clear_confirmation)
        
        for seg , content in segments.items():
            if seg not in st.session_state['developer-info-selected']: continue
            with expander_subheader(f'developer-info-{seg}' , seg , content['icon'] , height = H):
                content['operation']()
        
def main():
    set_current_page(PAGE_NAME)
    print_page_header(PAGE_NAME)
    show_developer_info()
    show_sidebar_buttons()
        
if __name__ == '__main__':
    main() 