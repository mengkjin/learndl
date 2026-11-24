#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 同步共享配置，更新数据和因子，续训到期生产模型并推算预测
# email: True
# mode: shell
# parameters:
#   forfeit_if_done:
#       type: bool
#       desc: skip when already finished today and source=bash; pass False from crontab to force rerun
#       required: False
#       default: True


from src.proj import CALENDAR
from src.proj.util.script import ScriptTool
from src.proj.util.filesys.shared_sync import SharedSync

@ScriptTool('daily_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(forfeit_if_done : bool = True , **kwargs):
    from src.api.pkgs.update import UpdateAPI
    SharedSync.sync()
    UpdateAPI.daily()
    import time
    from src.proj import PATH, MACHINE
    from src.api.pkgs.data import DataAPI
    from src.automation.artifact_exchange.storage_insights import publish_daily_safe
    if MACHINE.updatable and DataAPI.is_updated():
        from src.automation.artifact_exchange.production import sync_current_safe
        sync_current_safe()
        publish_daily_safe(PATH.production, time.time())

if __name__ == '__main__':
    main()
