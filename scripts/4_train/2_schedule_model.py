#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2025-10-01
# description: Train Schedule Model
# content: 训练某个计划好的模型,模型的参数在configs/schedule或.local_resources/shared/schedule_model里定义,也可以改变configs
# email: True
# mode: shell
# parameters:
#   schedule_name : 
#       type : Options.available_schedules()
#       desc : schedule config file to train
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
#   resume : 
#       type : [True , False]
#       prefix : "resume/"
#       default : "resume/False"
#       desc : resume training , if change schedule file, set to False
#       required : True
#   start : 
#       type : int
#       desc : start date
#   end : 
#       type : int
#       desc : end date
#   training_mode:
#       type: [auto, quick, sparse, full]
#       default: auto
#       desc: exploration mode; auto keeps saved settings, sparse only thins training nodes
#   training_step:
#       type: int
#       desc: exploration node stride (default 2)
#   quick_max_epoch:
#       type: int
#       desc: NN quick epoch cap (default 20; callbacks may increase it)
#   quick_window:
#       type: int
#       desc: NN quick training window cap (default 1200)
# file_editor:
#   name : "Schedule Config File Editor"
#   path: "{PATH.sched}/{schedule_name}.yaml | {PATH.sched_shared}/{schedule_name}.yaml"
#   height : 300 # optional

from pathlib import Path

from src.proj.util.script import ScriptTool

@ScriptTool('train_schedule_model' , '@schedule_name' , lock_num = 2 , markdown_catcher = True)
def main(schedule_name : str | None = None , short_test : bool | None = None , 
         resume : bool | None = None , start : int | None = None , end : int | None = None ,
         base_path : str | Path | None = None, resume_selection : str = 'interactive',
         training_mode: str | None = None, training_step: int | None = None,
         quick_max_epoch: int | None = None, quick_window: int | None = None,
         lifecycle_options: dict | None = None, **kwargs):
    from src.api.pkgs.model import ModelAPI
    from src.res.model.util.lifecycle import ModelLifecycle
    assert schedule_name is not None , 'schedule_name is required'
    options = dict(lifecycle_options or {})
    for key, value in (('step', training_step), ('max_epoch', quick_max_epoch), ('window', quick_window)):
        if value is not None:
            if key in options and options[key] != value:
                raise ValueError(f'Conflicting lifecycle option: {key}')
            options[key] = value
    options = ModelLifecycle.training_options(training_mode, options)
    ModelAPI.schedule_model(schedule_name , short_test , 0 if resume is None else int(resume) , start = start , end = end ,
                            resume_selection=resume_selection,
                            **({'lifecycle_options': options} if options else {}),
                            **({'base_path': Path(base_path)} if base_path else {}))
        
if __name__ == '__main__':
    main()
