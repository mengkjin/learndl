#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: 每周更新模型(只在服务器上)

import argparse , sys , pathlib , traceback
from datetime import datetime

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import ModelAPI
from src.basic import DualPrinter , send_email , AutoRunTask

def get_args():
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='not_specified', help='Source of the script call')
    parser.add_argument('--email', type=int, default=1, help='Send email or not')
    args , _ = parser.parse_known_args()
    return args

def main():
    with AutoRunTask('weekly update' , **AutoRunTask.get_args()) as runner:
        ModelAPI.update_models()
    '''
    args = get_args()
    print(f'Script Source: {args.source}')

    time_str = datetime.now().strftime("%Y%m%d")
    with DualPrinter(f'weekly_update.{time_str}.txt') as printer:
        try:
            ModelAPI.update_models()
            email_body = 'Successful Weekly Update!'
        except Exception as e:
            print(f'Error Occured! Info : ' + '-' * 20)
            print(e)

            print('Traceback : ' + '-' * 20)
            print(traceback.format_exc())
            email_body = f'Error Occured! {e}'
    if args.email: send_email(title = f'Weekly Update at {time_str}' , body = email_body , attachment = printer.filename)

    '''
    

if __name__ == '__main__':
    main()