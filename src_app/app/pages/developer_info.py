import streamlit as st

from src_app.frontend.frontend import ActionLogger

from util import SC , set_current_page , show_run_button_sidebar , print_page_header

PAGE_NAME = 'developer_info'

def developer_info_selected_change():
    selected = getattr(st.session_state , 'developer-info-selected' , [])
    if 'All' in selected:
        st.session_state['developer-info-selected'] = ['Session Control' , 'Session States' , 'Task Queue' , 'Action Logs' , 'Error Logs']
    if 'None' in selected:
        st.session_state['developer-info-selected'] = []
    

def show_developer_info(H = 500):
    """show developer info"""
    with st.container(key = "developer-info-special-expander"):
        st.segmented_control('developer-info-selected' , 
                                options = ["All" , "None" , "Session Control" , "Session States" , "Task Queue" , "Action Logs" , "Error Logs"] , 
                                key = "developer-info-selected" , 
                                selection_mode = "multi" ,
                                default = ["Session Control" , "Session States" , "Task Queue" , "Action Logs" , "Error Logs"] , label_visibility = "collapsed" ,
                                on_change = developer_info_selected_change)
        
        if "Session Control" in st.session_state['developer-info-selected']:
            with st.expander("Session Control" , expanded = False , icon = ":material/settings:").container(height = H):
                st.write(SC) 
            
        if "Session States" in st.session_state['developer-info-selected']:
            with st.expander("Session States" , expanded = False , icon = ":material/star:").container(height = H):
                st.write(st.session_state)
        
        if "Task Queue" in st.session_state['developer-info-selected']:
            with st.expander("Task Queue" , expanded = False , icon = ":material/directory_sync:").container(height = H):
                SC.task_queue.refresh()     
                st.json(SC.task_queue.queue_content() , expanded = 1)
        
        if "Action Logs" in st.session_state['developer-info-selected']:
            with st.expander("Action Logs" , expanded = False , icon = ":material/format_list_numbered:"):
                st.code(ActionLogger.get_action_log(), language='log' , height = H , wrap_lines = True)
        
        if "Error Logs" in st.session_state['developer-info-selected']:
            with st.expander("Error Logs" , expanded = False , icon = ":material/error:"):
                st.code(ActionLogger.get_error_log(), language='log' , height = H , wrap_lines = True)

        cols = st.columns(7) # 7 buttons in a row
            
        with cols[0]:
            st.button("Log" , icon = ":material/delete_forever:" , key = "developer-log-clear" , 
                      help = "Clear Both Action and Error Logs" ,
                      on_click = SC.click_log_clear_confirmation)
        
def main():
    set_current_page(PAGE_NAME)
    print_page_header(PAGE_NAME)
    show_developer_info()
    show_run_button_sidebar()
        
if __name__ == '__main__':
    main() 