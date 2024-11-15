import sys , pathlib , traceback
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

def main_process():
    try:
        print('update_data: ' + '*' * 20)
        DataAPI.update()

        print('prepare_predict_data: ' + '*' * 20)
        DataAPI.prepare_predict_data()

        print('update_models: ' + '*' * 20)
        ModelTrainer.update_models()

        print('update_hidden: ' + '*' * 20)
        ModelHiddenExtractor.update_hidden()

        print('update_factors: ' + '*' * 20)
        ModelPredictor.update_factors()

    except Exception as e:
        print(f'Error Occured!')

        print('Error Info : ' + '*' * 20)
        print(e)

        print('Traceback : ' + '*' * 20)
        print(traceback.format_exc())

if __name__ == '__main__':
    # proceed_new_version()    
    with DualPrinter() as printer:
        main_process()
    send_email(title = email_title , body = printer.contents())
    
    