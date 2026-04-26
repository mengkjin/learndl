"""Task queue page: full task list with filter controls, pagination, and inline reports."""
from src.interactive.main.util import SC
from src.interactive.main.util.components import show_task_queue

PAGE_NAME = 'task_queue'

@SC.wrap_page(PAGE_NAME)
def main() -> None:
    """Entry point for the task queue page."""
    show_task_queue()

if __name__ == '__main__':
    main() 
    