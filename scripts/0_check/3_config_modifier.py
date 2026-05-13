#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-05-09
# description: modify model configs
# content: modify model configs
# email: True
# mode: shell

from src.proj.util import ScriptTool
from src.res.model.util.config import ModelConfigsInspector , ModelConfigsBatchModifier

@ScriptTool('config_modifier')
def main(**kwargs):     
    modifier = ModelConfigsBatchModifier()
    modifier.batch_modify()
    inspecter = ModelConfigsInspector()
    inspecter.inspect_key_values({'ResetOptimizer' : False , 'lamb' : True , 'eps' : True , 'EarlyExitRetrain' : False})

if __name__ == '__main__':
    main()