import streamlit as st
import re
from typing import Literal

from src_app.db import RUNS_DIR

from util import starter , SC , runs_page_url

def show_task_queue(queue_type : Literal['full' , 'filter' , 'latest'] = 'filter'):
    container = st.container(key="task-queue-special-expander")
    with container:
        st.header(":material/event_list: Running Queue" , divider = 'grey')
        st.info("Shows the entire running queue" , icon = ":material/info:")
        st.info("Tailor filters to show exact tasks" , icon = ":material/info:")
        
        show_queue_header()
        if queue_type == 'filter': show_queue_filters()
        show_queue_item_list(queue_type)

def show_queue_header():
    with st.container(key = "queue-header-buttons"):
        cols = st.columns([1, 1 , 1 , 5] , gap = "small" , vertical_alignment = "center")
        with cols[0]:  
            st.button(":material/directory_sync:", key="task-queue-sync",  
                        help = "Sync Historical Tasks into Current Queue" ,
                        on_click = SC.click_queue_sync)
        with cols[1]:  
            st.button(":material/refresh:", key="task-queue-refresh",  
                        help = "Refresh Queue" ,
                        on_click = SC.click_queue_refresh)
            
        with cols[2]:
            st.button(":material/delete:", key="task-queue-empty", 
                        help = "Empty Queue" ,
                        on_click = SC.click_queue_empty)
            
        with cols[3]:
            st.button(":material/delete_forever:", key="task-queue-clear", 
                        help = "Clear Queue" ,
                        on_click = SC.click_queue_clear_confirmation)
            
    if SC.queue_last_action:
        if SC.queue_last_action[1]:
            st.success(SC.queue_last_action[0] , icon = ":material/check_circle:")
        else:
            st.error(SC.queue_last_action[0] , icon = ":material/error:")
        
    if SC.task_queue.is_empty():
        st.info("Queue is empty, click the script below to run and it will be displayed here" , icon = ":material/queue_play_next:")
        return

    st.caption(f":rainbow[:material/bar_chart:] {SC.task_queue.status_message()}")
        
def show_queue_filters():
    with st.container(key="queue-filter-container").expander("Queue Filters" , expanded = True , icon = ":material/filter_list:"):
        status_options = ["All" , "Running" , "Complete" , "Error"]
        folder_options = [item.path for item in SC.path_items if item.is_dir]
        file_options = [item.path for item in SC.path_items if item.is_file]
        st.radio(":gray-badge[**Running Status**]" , status_options , key = "queue-filter-status", horizontal = True ,
                    on_change = SC.click_queue_filter_status)
        st.multiselect(":gray-badge[**Script Folder**]" , folder_options , key = "queue-filter-path-folder" ,
                        format_func = lambda x: str(x.relative_to(RUNS_DIR)) ,
                        on_change = SC.click_queue_filter_path_folder)
        st.multiselect(":gray-badge[**Script File**]" , file_options , key = "queue-filter-path-file" ,
                        format_func = lambda x: x.name ,
                        on_change = SC.click_queue_filter_path_file)
        
def show_queue_item_list(queue_type : Literal['full' , 'filter' , 'latest'] = 'latest'):
    if queue_type == 'full':
        queue = SC.task_queue
        container_height = 500
    elif queue_type == 'filter':
        queue = SC.filter_task_queue()
        container_height = None
    elif queue_type == 'latest':
        queue = SC.latest_task_queue()
        container_height = None
        st.info(f"Showing latest {len(queue)} tasks" , icon = ":material/info:")

    with st.container(height = container_height , key = f"queue-item-list-container"):
        for item in queue.values():
            placeholder = st.empty()
            container = placeholder.container(key = f"queue-item-container-{item.id}")
            with container:
                cols = st.columns([18, 1 , 1] , gap = "small" , vertical_alignment = "center")
                    
                help_text = '|'.join([f"Status: {item.status}" , f"Dur: {item.duration_str}", f"PID: {item.pid}"])
                cols[0].button(f"{item.tag_icon} {item.button_str}",  help=help_text , key=f"queue-item-content-{item.id}" , 
                            use_container_width=True , on_click = SC.click_queue_item , args = (item,))
                
                cols[1].button(":red-badge[:material/cancel:]", 
                            key=f"queue-item-remover-{item.id}", help="Remove/Terminate", 
                            on_click = SC.click_queue_remove_item , args = (item,))
                
                if cols[2].button(
                    ":blue-badge[:material/slideshow:]", 
                    key=f"show-complete-report-{item.id}" ,
                    help = "Show complete report in main page" ,
                    on_click = SC.click_show_complete_report , args = (item,)):
                    st.switch_page(runs_page_url(str(item.relative)))
                
                if SC.running_report_queue is None or SC.running_report_queue != item.id:
                    continue
            
                status_text = f'Running Report {item.status_state.title()}'
                status = st.status(status_text , state = item.status_state , expanded = True)

                with status:
                    col_config = {
                        'Item': st.column_config.TextColumn(width=None, help='Key of the item'),
                        'Value': st.column_config.TextColumn(width="large", help='Value of the item')
                    }

                    st.dataframe(item.dataframe() , row_height = 20 , column_config = col_config)
                    SC.wait_for_complete(item)
                    if item.status == 'complete':
                        st.success(f'Script Completed' , icon = ":material/add_task:")
                    elif item.status == 'error':
                        st.error(f'Script Failed' , icon = ":material/error:")
            
if __name__ == '__main__':
    starter()
    show_task_queue() 
    