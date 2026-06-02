#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: codex
# date: 2026-04-30
# description: Test Announcement Race Mode
# content: 测试公告异步竞速模式，观测winner、取消与temp清理行为
# email: True
# mode: shell

from src.proj import Proj
from src.proj.util.script import ScriptTool
from src.data.crawler.announcement.agent import AnnouncementAgent

@ScriptTool("test_announcement_race_mode")
def main():
    Proj.debug.start()
    AnnouncementAgent.update_all('update', force_update=2)

if __name__ == "__main__":
    main()
