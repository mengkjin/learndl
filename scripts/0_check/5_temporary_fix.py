#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.data.crawler.announcement.agent import AnnouncementAgent

@ScriptTool('temporary_fix')
def main(**kwargs):
    AnnouncementAgent.parallel_run(20220101,20260318,workers=10)
        
if __name__ == '__main__':
    main()
        
    