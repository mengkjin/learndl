"""Per-script detail page: task selector, parameter form, and report viewer.

This module is the main content of every auto-generated script wrapper page.
:func:`show_script_detail` is the single entry point: it renders the page
header, historical-task selector, parameter-settings form, and the running /
completed task report in sequence.
"""

from .session_control import SC
from .components import show_task_history , show_param_settings , show_report_main

__all__ = ['show_script_detail']

def script_detail_layout(script_key : str):
    """show script detail"""
    show_task_history(script_key)
    show_param_settings(script_key)
    show_report_main(script_key)

def show_script_detail(script_key : str):
    """show main part"""
    SC.wrap_page(script_key)(script_detail_layout)(script_key)