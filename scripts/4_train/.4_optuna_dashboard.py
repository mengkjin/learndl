#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-06-02
# description: Launch Optuna Dashboard
# content: Launch Optuna Dashboard
# email: False
# mode: shell

from src.api.pkgs.dashboard import DashboardAPI

if __name__ == '__main__':
    DashboardAPI.optuna_dashboard()
    