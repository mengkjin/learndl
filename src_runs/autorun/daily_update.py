import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import DataAPI , ModelTrainer , ModelPredictor , ModelHiddenExtractor
from src.basic.util.logger import dual_printer , DualPrinter
from src.basic.util.email import email_myself , send_email
from datetime import datetime

email_title = f'daily update at {datetime.now().strftime("%Y-%m-%d")}'

@email_myself(True , title = email_title)
@dual_printer
def proceed_new_version():
    return main_process()

def proceed_old_version():
    with DualPrinter() as printer:
        main_process()
    send_email(title = email_title , body = printer.contents())

def main_process():
    try:
        DataAPI.update()
    except Exception as e:
        print(f'DataAPI went wrong: {e}')

    print('prepare_predict_data')
    DataAPI.prepare_predict_data()
    ModelTrainer.update_models()
    ModelHiddenExtractor.update_hidden()
    ModelPredictor.update_factors()

if __name__ == '__main__':
    # proceed_new_version()    
    proceed_old_version()
    
    