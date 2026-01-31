#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-01-29
# description: test email application
# content: very short script to test email sending 
# email: True
# mode: shell

from src.proj.util import Email
from src.proj.util import ScriptTool 

@ScriptTool('test_email' , lock_timeout = 10)
def main(**kwargs):
    Email.SETTINGS.print_info()
        
if __name__ == '__main__':
    main()
        
    