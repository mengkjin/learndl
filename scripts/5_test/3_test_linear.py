#!/usr/bin/env python
# author: jinmeng
# date: 2026-10-12
# description: Test Linear Model
# content: 测试 configs/schedule/linear 中定义的线性组合，默认重新测试，显式 resume 才续测
# email: True
# mode: shell
# parameters:
#   schedule_name:
#       type: Options.available_linear_schedules()
#       desc: choose a linear schedule
#       required: True
#   resume:
#       type: [True, False]
#       prefix: "resume/"
#       default: "resume/False"
#       desc: resume testing; use False after changing the configuration
#   start:
#       type: int
#       desc: start date
#   end:
#       type: int
#       desc: end date
# file_editor:
#   name: "Linear Schedule Config File Editor"
#   path: "{PATH.sched_linear}/{schedule_name}.yaml"
#   height: 300

from src.proj import Options
from src.proj.util.script import ScriptTool
from src.res.model.model_module.application import ModelTrainer


@ScriptTool('test_linear', '@schedule_name', lock_num=0)
def main(schedule_name: str | None = None, resume: bool = False,
         start: int | None = None, end: int | None = None, **kwargs):
    if schedule_name not in Options.available_linear_schedules(refresh=True):
        raise ValueError(f'A schedule in configs/schedule/linear is required: {schedule_name!r}')
    return ModelTrainer.schedule(schedule_name, short_test=False, resume=int(bool(resume)), start=start, end=end)


if __name__ == '__main__':
    main()
