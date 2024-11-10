from src.api import DataAPI , ModelPredictor , ModelHiddenExtractor
from src.basic.util.logger import dual_printer 
from src.basic.util.email import email_myself
from datetime import datetime

@email_myself(False , title = f'daily update at {datetime.now().strftime("%Y-%m-%d")}')
@dual_printer
def proceed():
    try:
        DataAPI.update()
    except Exception as e:
        print(f'DataAPI went wrong: {e}')

    print('prepare_predict_data')
    DataAPI.prepare_predict_data()

    ModelHiddenExtractor.update_hidden('gru_day' , model_submodels=['best'])
    ModelPredictor.update_factors()