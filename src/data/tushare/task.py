from .basic import TushareFetecher
from .download import TUSHARE_DOWNLOAD_TASK

from .model import TuShareCNE5_Calculator

def main():
    for task_down in TUSHARE_DOWNLOAD_TASK:
        task = task_down()
        assert isinstance(task , TushareFetecher) , task
        task.update()

    task_cne5 = TuShareCNE5_Calculator()
    task_cne5.Update(job_list=['exposure'])
    task_cne5.Update(job_list=['risk'])