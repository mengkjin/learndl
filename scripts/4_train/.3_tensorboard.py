#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2026-03-18
# description: Launch TensorBoard for a Model
# content: Launch TensorBoard for a model
# email: False
# mode: shell

from src.api.pkgs.dashboard import DashboardAPI

if __name__ == '__main__':
    DashboardAPI.tensorboard()
    