import sys , pathlib , traceback
from datetime import datetime
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import DataAPI , ModelAPI
from src.basic import ProcessAndEmail , DualPrinter , send_email

email_title = f'daily update at {datetime.now().strftime("%Y-%m-%d")}'

@ProcessAndEmail('daily_update' , email_title)
def update_main():
    DataAPI.reconstruct_train_data()
    ModelAPI.update_models()


if __name__ == '__main__':
    # proceed_new_version()
    # 
    # 
    # 
    # update_main()
        
    with DualPrinter('daily_update.txt') as printer:
        try:
            DataAPI.reconstruct_train_data()
            ModelAPI.update_models()

        except Exception as e:
            print(f'Error Occured!')

            print('Error Info : ' + '*' * 20)
            print(e)

            print('Traceback : ' + '*' * 20)
            print(traceback.format_exc())
    send_email(title = email_title , body = printer.contents())