import argparse , traceback
from datetime import datetime

from .logger import DualPrinter
from .email import send_email
    
class AutoRunTask:
    def __init__(self , task_name : str , source : str | None = None , email = True , **kwargs):
        self.task_name = task_name.replace(' ' , '_')
        self.source = source
        self.email = email

    def __enter__(self):
        if self.source: print(f'Script Source: {self.source}')
        self.time_str = datetime.now().strftime('%Y%m%d')
        self.printer = DualPrinter('.'.join([self.task_name , self.time_str , 'txt']))
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
        if self.email: 
            title = ' '.join([*[s.capitalize() for s in self.task_name.split('_')] , 'at' , self.time_str])
            send_email(title = title , body = self.status , attachment = self.printer.filename)

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description='Run daily update script.')
        parser.add_argument('--source', type=str, default='not_specified', help='Source of the script call')
        parser.add_argument('--email', type=int, default=1, help='Send email or not')
        parser.add_argument('--param', type=str, default='', help='Extra parameters for the script')
        args , _ = parser.parse_known_args()
        return args.__dict__


