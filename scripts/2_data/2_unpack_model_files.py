#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2026-01-18
# description: Unpack Model Files
# content: 解包模型文件
# blacklist:
#   machine: ['mengkjin-server']
# email: True
# mode: shell


from src.proj.util.script import ScriptTool

from src.res.model.util import PredictorPath

@ScriptTool('unpack_model_files')
def main(**kwargs):
    PredictorPath.UnpackModelArchives()
    
if __name__ == '__main__':
    main()
