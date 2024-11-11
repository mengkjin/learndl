import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.api import DataAPI , ModelPredictor , ModelHiddenExtractor
from src.basic import PATH
from src.basic.util.logger import dual_printer 
from src.basic.util.email import email_myself , send_email
from datetime import datetime

def update_process():
    try:
        DataAPI.update()
    except Exception as e:
        print(f'DataAPI went wrong: {e}')

    print('prepare_predict_data')
    DataAPI.prepare_predict_data()

    ModelHiddenExtractor.update_hidden('gru_day' , model_submodels=['best'])
    ModelPredictor.update_factors()

@email_myself(True , title = f'daily update at {datetime.now().strftime("%Y-%m-%d")}' , body_file = 'print_log.txt')
@dual_printer
def proceed_new_version():
    update_process()


class DualPrinter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass


if __name__ == '__main__':
    # proceed_new_version()

    target_filename = PATH.logs.joinpath('print_log.txt')
    
    sys.stdout = DualPrinter(target_filename)
    update_process()
    sys.stdout = sys.stdout.terminal

    with open(target_filename , 'r') as f:
        email_body = f.read()

    send_email(title = f'daily update at {datetime.now().strftime("%Y-%m-%d")}' , 
               body = email_body , 
               recipient_email = 'mengkjin@163.com')
    
    