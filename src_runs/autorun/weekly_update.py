#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: 每周更新模型(只在服务器上)

import sys , pathlib , traceback
from datetime import datetime

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import ModelAPI
from src.basic import DualPrinter , send_email

if __name__ == '__main__':
    time_str = datetime.now().strftime("%Y%m%d")
    email_title = f'Weekly Update at {time_str}'
    with DualPrinter(f'weekly_update.{time_str}.txt') as printer:
        try:
            ModelAPI.update_models()

        except Exception as e:
            print(f'Error Occured!')

            print('Error Info : ' + '-' * 20)
            print(e)

            print('Traceback : ' + '-' * 20)
            print(traceback.format_exc())
    send_email(title = email_title , body = printer.contents())