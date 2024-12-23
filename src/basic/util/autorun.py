import traceback
from datetime import datetime

from .logger import DualPrinter
from .email import Email
    
class AutoRunTask:
    def __init__(self , task_name : str , email = True , email_if_attachment = True ,**kwargs):
        self.task_name = task_name.replace(' ' , '_')
        self.email = email
        self.email_if_attachment = email_if_attachment

    def __enter__(self):
        self.date_str = datetime.now().strftime('%Y%m%d')
        self.time_str = datetime.now().strftime('%H%M%S')
        name = '.'.join([self.task_name , f'{self.date_str}_{self.time_str}' , 'txt'])
        self.printer = DualPrinter(f'{self.task_name}/{name}')
        self.printer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            print(f'Error Occured! Info : ' + '-' * 20)
            print(exc_value)

            print('Traceback : ' + '-' * 20)
            print(traceback.format_exc())
            self.status = f'Error Occured! {exc_value}'
        else:
            self.status = ' '.join(['Successful' , *[s.capitalize() for s in self.task_name.split('_')]]) + '!'
        self.printer.__exit__(exc_type, exc_value, exc_traceback)
        if self.email or (self.email_if_attachment and Email.ATTACHMENTS): 
            title = ' '.join([*[s.capitalize() for s in self.task_name.split('_')] , 'at' , self.date_str])
            Email.attach(self.printer.filename)
            Email().send(title = title , body = self.status)