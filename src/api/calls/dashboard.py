"""
Direct calls related to dashboard management of this project.
1. Tensorboard
2. Optuna Dashboard
"""
from __future__ import annotations

from src.api.util.direct_call import DirectCall

__all__ = [
    'Tensorboard' , 'OptunaDashboard' , 
]

# %% project code related operations
class Tensorboard(DirectCall):
    """Launch Tensorboard."""
    category = 'Dashboard'
    def run(self) -> None:
        from src.api.pkgs.dashboard import DashboardAPI
        DashboardAPI.tensorboard()

class OptunaDashboard(DirectCall):
    """Launch Optuna Dashboard."""
    category = 'Dashboard'
    def run(self) -> None:
        from src.api.pkgs.dashboard import DashboardAPI
        DashboardAPI.optuna_dashboard()