import streamlit as st
import re
from typing import Literal

from src_app.db import RUNS_DIR

from util import SC , runs_page_url , set_current_page , show_run_button_sidebar , print_page_header

PAGE_NAME = 'task_queue'

def on_first_page(max_page : int):
    if st.session_state.get('choose-task-page') == 1: return
    st.session_state['choose-task-page'] = 1
def on_last_page(max_page : int):
    if st.session_state.get('choose-task-page') == max_page: return
    st.session_state['choose-task-page'] = max_page
def on_prev_page(max_page : int):
    if st.session_state.get('choose-task-page') == 1: return
    st.session_state['choose-task-page'] = max((st.session_state.get('choose-task-page') or 1) - 1, 1)
def on_next_page(max_page : int):
    if st.session_state.get('choose-task-page') == max_page: return
    st.session_state['choose-task-page'] = (st.session_state.get('choose-task-page') or 1) + 1


def show_task_queue(queue_type : Literal['full' , 'filter' , 'latest'] = 'filter'):
    with st.container(key="task-queue-special-expander"):
        show_queue_header()
        if queue_type == 'filter': show_task_filters()
        show_queue_item_list(queue_type)

def show_queue_header():
    with st.container(key = "queue-header-buttons"):
        buttons = {
            'sync' : {
                'call' : SC.click_queue_sync , 
                'icon' : ":material/directory_sync:" ,
                'help' : "Sync Historical Tasks into Current Queue" ,
            } ,
            'refresh' : {
                'call' : SC.click_queue_refresh , 
                'icon' : ":material/refresh:" ,
                'help' : "Refresh Queue" ,
            } ,
            'clean' : {
                'call' : SC.click_queue_clean , 
                'icon' : ":material/mop:" ,
                'help' : "Remove All Error Tasks (Irreversible)" ,
            } ,
            'delist-all' : {
                'call' : SC.click_queue_delist_all , 
                'icon' : ":material/clear_all:" ,
                'help' : "Delist All Tasks in Queue" ,
            } ,
            'remove-all' : {
                'call' : SC.click_queue_remove_all , 
                'icon' : ":material/delete_history:" ,
                'help' : "Backup and Remove All" ,
            } ,
            'restore-all' : {
                'call' : SC.click_queue_restore_all , 
                'icon' : ":material/restore:" ,
                'help' : "Restore from Backup" ,
            } ,
        }
        cols = st.columns(min(len(buttons) * 2 , 10) , gap = "small" , vertical_alignment = "center")
        
        for col , (name , but) in zip(cols[:len(buttons)] , buttons.items()):
            col.button(but['icon'], key=f"task-queue-{name}",  help = but['help'] , on_click=but['call'])
            
    if SC.queue_last_action:
        if SC.queue_last_action[1]:
            st.success(SC.queue_last_action[0] , icon = ":material/check_circle:")
        else:
            st.error(SC.queue_last_action[0] , icon = ":material/error:")
        
    if SC.task_queue.is_empty():
        st.warning("Queue is empty, click the script below to run and it will be displayed here" , icon = ":material/queue_play_next:")
        return

    st.caption(f":rainbow[:material/bar_chart:] {SC.task_queue.status_message()}")
    st.caption(f":rainbow[:material/bar_chart:] {SC.task_queue.source_message()}")
        
def show_task_filters():
    with st.container(key="task-filter-container").expander("Task Filters" , expanded = True , icon = ":material/filter_list:"):
        status_options = ["All" , "Running" , "Complete" , "Error"]
        source_options = ["All" , "Py" , "App" ,"Bash" , "Other"]
        folder_options = [item.path for item in SC.path_items if item.is_dir]
        file_options = [item.path for item in SC.path_items if item.is_file]
        st.radio(":blue-badge[**Running Status**]" , status_options , key = "task-filter-status", horizontal = True ,
                 on_change = SC.click_queue_filter_status)
        st.radio(":blue-badge[**Script Source**]" , source_options , key = "task-filter-source", horizontal = True ,
                    on_change = SC.click_queue_filter_source)
        st.multiselect(":blue-badge[**Script Folder**]" , folder_options , key = "task-filter-path-folder" ,
                        format_func = lambda x: str(x.relative_to(RUNS_DIR)) ,
                        on_change = SC.click_queue_filter_path_folder)
        st.multiselect(":blue-badge[**Script File**]" , file_options , key = "task-filter-path-file" ,
                        format_func = lambda x: x.name ,
                        on_change = SC.click_queue_filter_path_file)
        
def show_queue_item_list(queue_type : Literal['full' , 'filter' , 'latest'] = 'latest'):
    if queue_type == 'full':
        queue = SC.task_queue.queue
        container_height = 500
    elif queue_type == 'filter':
        queue = SC.get_filtered_queue()
        container_height = None
    elif queue_type == 'latest':
        queue = SC.get_latest_queue()
        container_height = None
        st.info(f"Showing latest {len(queue)} tasks" , icon = ":material/info:")

    item_ids = list(queue.keys())
    if SC.running_report_queue is not None and SC.running_report_queue in item_ids:
        choose_index = item_ids.index(SC.running_report_queue)
    else:
        choose_index = None

    page_option_cols = st.columns(2)

    with page_option_cols[0].container(key = f"choose-task-num-per-page-container"):
        cols = st.columns(2 , vertical_alignment = "center")
        cols[0].info('**Tasks per Page**')
        num_per_page = cols[1].selectbox('select num per page', [20 , 50 , 100 , 500], format_func = lambda x: f'{x} Tasks/Page', 
                            index = 0 , key = f"choose-task-item-num-per-page" , label_visibility = 'collapsed')
        
    with page_option_cols[1].container(key = f"choose-task-page-container"):
        max_page = (len(item_ids) - 1) // num_per_page + 1
        page_options = list(range(1, max_page + 1))
        index_page = choose_index // num_per_page if choose_index is not None else 0
        page_cols = st.columns([7, 1, 1, 3, 1, 1] , vertical_alignment = "center")
        page_cols[0].info(f'**Select Page (1 ~ {max_page})**')
        page_cols[1].button(":material/first_page:", key = f"choose-task-page-first", on_click = on_first_page , args = (max_page,))
        page_cols[2].button(":material/chevron_left:", key = f"choose-task-page-prev", on_click = on_prev_page , args = (max_page,))
        page_cols[3].selectbox('select page', page_options, key = f"choose-task-page" , index = index_page , 
                                placeholder = f'Page #',
                                format_func = lambda x: f'Page {x}', label_visibility = 'collapsed')
        page_cols[4].button(":material/chevron_right:", key = f"choose-task-page-next", on_click = on_next_page , args = (max_page,))
        page_cols[5].button(":material/last_page:", key = f"choose-task-page-last", on_click = on_last_page , args = (max_page,))
        
    with st.container(height = container_height , key = f"queue-item-list-container"):
        current_page = st.session_state.get('choose-task-page') or 1
        item_options = item_ids[(current_page - 1) * num_per_page : current_page * num_per_page]
        
        for i , item_id in enumerate(item_options):
            item = queue[item_id]
            index = (current_page - 1) * num_per_page + i + 1
            placeholder = st.empty()
            container = placeholder.container(key = f"queue-item-container-{item.id}")
            with container:
                cols = st.columns([18, .5,.5,.5,.5] , gap = "small" , vertical_alignment = "center")
                    
                help_text = ' | '.join([f"Status: {item.status.title()}" , 
                                      f"Source: {item.source.title()}" , 
                                      f"Dur: {item.duration_str}" , 
                                      f"PID: {item.pid}"])
                cols[0].button(f"{item.tag_icon} {index: <2}. {item.button_str_long}",  help=help_text , key=f"click-content-{item.id}" , 
                               use_container_width=True , on_click = SC.click_queue_item , args = (item,))
                if cols[1].button(
                    ":blue-badge[:material/slideshow:]", 
                    key=f"queue-item-report-{item.id}" ,
                    help = "Show complete report in main page" ,
                    on_click = SC.click_show_complete_report , args = (item,)):
                    st.switch_page(runs_page_url(str(item.relative)))
                    
                cols[3].button(":violet-badge[:material/remove:]", 
                            key=f"queue-item-delist-{item.id}", help="Delist from Queue", 
                            on_click = SC.click_queue_delist_item , args = (item,))
                cols[4].button(":red-badge[:material/cancel:]", 
                            key=f"queue-item-remove-{item.id}", help="Delete from Database (Irreversible)", 
                            on_click = SC.click_queue_remove_item , args = (item,))
                
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
                    SC.wait_until_completion(item)
                    if item.status == 'complete':
                        st.success(f'Script Completed' , icon = ":material/add_task:")
                    elif item.status == 'error':
                        st.error(f'Script Failed' , icon = ":material/error:")
            
def main():
    set_current_page(PAGE_NAME)
    print_page_header(PAGE_NAME)
    show_task_queue() 
    show_run_button_sidebar()

if __name__ == '__main__':
    main() 
    