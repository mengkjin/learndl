"""The control panel for the interactive app, will show at the top of every page.

Key objects:

* :class:`ControlPanelButton` / :class:`ControlPanel` — the shared action
  bar rendered at the top of every page via :meth:`SessionControl.get_control_panel`.
"""

from .panel import ControlPanel