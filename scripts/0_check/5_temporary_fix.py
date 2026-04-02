#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.api import UpdateAPI
from src.data.crawler.announcement.agent import AnnouncementAgent
from src.data.util.classes import DataCache

@ScriptTool('temporary_fix')
def main(**kwargs):
    DataCache.purge_all(confirm = True)
    # AnnouncementAgent.update()
    UpdateAPI.rollback(rollback_date = 20260327)
        
if __name__ == '__main__':
    main()
        
    