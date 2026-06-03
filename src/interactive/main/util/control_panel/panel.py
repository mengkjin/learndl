"""The control panel for the interactive app, will show at the top of every page.

Key objects:

* :class:`ControlPanelButton` / :class:`ControlPanel` — the shared action
  bar rendered at the top of every page via :meth:`SessionControl.get_control_panel`.
"""
from __future__ import annotations
import streamlit as st

from src.proj import Const

from . import buttons
from . import popovers


class ControlPanel:
    """Horizontal action bar rendered at the top of every app page.

    Contains a fixed set of :class:`ControlPanelButton` instances plus a
    settings popover for global run toggles (verbosity, email, silent mode).
    """
    control_panel_key = "control-panel"
    buttons : dict[str, buttons.ControlPanelButton] = {
        'script-runner-run' : buttons.ScriptRunnerRunButton(),
        'control-git-clear-pull' : buttons.ControlGitClearPullButton(),
        'control-pull-and-run' : buttons.ControlGitClearPullRunButton(),
        'control-refresh-interactive' : buttons.ControlRefreshInteractiveButton(),
    }
    popovers : dict[str, popovers.ControlPanelPopover] = {
        'intro-page' : popovers.IntroPagePopover(),
        'global-settings' : popovers.GlobalSettingsPopover(),
        'more-buttons' : popovers.MoreButtonsPopover(),
        'system-info' : popovers.SystemInfoPopover(),
    }

    @property
    def area_columns(self) -> tuple[float, float, float]:
        width_ratio = Const.Pref.interactive.get('control_panel_width_ratio' , 0.8)
        return (1 - width_ratio) / 2 , width_ratio , (1 - width_ratio) / 2
    
    def show(self , script_key : str | None = None) -> None:
        """Render the full control panel (buttons + settings popover).

        Args:
            script_key: Passed through to each button so they can
                enable/disable themselves based on whether a script is active.
        """
        with st.container(key = f"{self.control_panel_key}-container"):
            self.show_buttons(script_key = script_key)
            self.show_popovers(script_key = script_key)

    def show_buttons(self , script_key : str | None = None) -> None:
        """Render the control panel buttons area.

        Args:
            script_key: Passed through to each button so they can
                enable/disable themselves based on whether a script is active.
        """
        _ , area , _ = st.columns(self.area_columns , gap = 'small' , vertical_alignment = 'center')
        min_cols , max_cols = Const.Pref.interactive.get('control_panel_buttons_columns' , [3 , 5])
        with area.container(key = f"{self.control_panel_key}-buttons"):
            nrows = (len(self.buttons) / max_cols).__ceil__()
            ncols = max(min(len(self.buttons) , max_cols) , min_cols)
            buttons = list(self.buttons.values())
            for irow in range(nrows):
                cols = st.columns(ncols , gap = 'small' , vertical_alignment = 'center')
                for icol , col in zip(range(ncols), cols):
                    if irow * ncols + icol >= len(buttons):
                        break
                    button = buttons[irow * ncols + icol]
                    with col:
                        button.show(script_key = script_key)

    def show_popovers(self , script_key : str | None = None) -> None:
        """Render the control panel popovers area.
        """
        _ , area , _ = st.columns(self.area_columns , gap = 'small' , vertical_alignment = 'center')
        min_cols , max_cols = Const.Pref.interactive.get('control_panel_popovers_columns' , [2 , 4])

        with area.container(key = f"{self.control_panel_key}-popovers" , horizontal_alignment = 'center'):
            nrows = (len(self.popovers) / max_cols).__ceil__()
            ncols = max(min(len(self.popovers) , max_cols) , min_cols)
            popovers = list(self.popovers.values())
            for irow in range(nrows):
                cols = st.columns(ncols , gap = 'small' , vertical_alignment = 'center')
                for icol , col in zip(range(ncols), cols):
                    if irow * ncols + icol >= len(popovers):
                        break
                    popover = popovers[irow * ncols + icol]
                    with col:
                        popover.show(script_key = script_key)