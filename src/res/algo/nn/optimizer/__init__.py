"""SAM-family sharpness-aware minimizers.

Available optimizers:
    SAM         — Sharpness-Aware Minimization (Foret et al. 2021)
    SSAMF       — Sparse SAM with Fisher Information mask (Liu et al. 2022)
    ASAM        — Adaptive SAM with weight-scaled perturbation (Kim et al. 2021)
    GSAM        — Gradient SAM with gradient decomposition (Zhuang et al. 2022)
    GAM         — Gradient Agreement Maximization (3-forward-pass)
    FriendlySAM — Momentum-corrected perturbation (Zhang et al. 2023)
"""
from .sam import SAM , SSAMF , ASAM , GSAM , GAM , FriendlySAM
