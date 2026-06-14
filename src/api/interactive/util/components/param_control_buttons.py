"""
param control buttons
a horizontal row of buttons for resetting and last parameters
"""
from __future__ import annotations
from src.api.util.backend import ScriptRunner

from src.api.util.st_frontend import SacBoundButton , SacOnClickButtons
from src.api.interactive.util.session_control import SC

__all__ = [
    'param_control_buttons'
]

def param_control_buttons(runner : ScriptRunner) -> None:
    """Callback: jump to the first page of the task list.

    Args:
        max_page: Total number of pages (unused but required by Streamlit callback).
    """
    def _on_reset_params() -> None:
        SC.script_params_cache.clear_cache(runner.script_key)
        SC.current_task_item = None
        SC.param_inputs_form.reset_options()

    def _on_last_params() -> None:
        item = SC.get_latest_task_item(runner.script_key)
        if item is not None:
            item_params = SC.param_inputs_form.cmd_to_param_values(cmd = item.cmd)
            SC.script_params_cache.update_cache(runner.script_key, 'value', item_params)
            SC.param_inputs_form.reset_options()

    SacOnClickButtons(
        [
            SacBoundButton(label = 'Reset Parameters' , icon = 'c-circle-fill' , on_click = _on_reset_params),
            SacBoundButton(label = 'Last Parameters' , icon = 'rewind-circle-fill' , on_click = _on_last_params),
        ],
        key = f"param-inputs-form-sac-actions-{runner.script_key}",
        size = 'xs' , variant = 'filled' , color = '#25C3B0'
    ).render()
