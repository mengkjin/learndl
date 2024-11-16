import traceback
from datetime import datetime
from .logger import DualPrinter
from .email import send_email

class ProcessAndEmail:
    def __init__(self , project_name : str , email_title : str | None = None):
        # title = f'daily update at {datetime.now().strftime("%Y-%m-%d")}'
        self.project_name = project_name
        self.email_title = email_title if email_title is not None else f'{project_name} at {datetime.now().strftime("%Y-%m-%d")}'
    
    def __call__(self , func):
        def wrapper(*args , **kwargs):
            with DualPrinter(f'{self.project_name}.txt') as printer:
                try:
                    ret = func(*args , **kwargs)
                except Exception as e:
                    print(f'Error Occured!')
                    print('Error Info : ' + '*' * 20)
                    print(e)
                    print('Traceback : ' + '*' * 20)
                    print(traceback.format_exc())

            send_email(title = self.email_title , body = printer.contents())
            return ret
        return wrapper
