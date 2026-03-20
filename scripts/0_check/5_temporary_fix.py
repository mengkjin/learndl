#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.api import ModelAPI
from src.res.model.util.config import ModelConfig

@ScriptTool('temporary_fix')
def main(**kwargs):
    models = ModelAPI.available_models(True , True)
    for model in models:
        config = ModelConfig(model , stage = 2 , resume = 1 , selection = 0)
        
if __name__ == '__main__':
    main()
        
    