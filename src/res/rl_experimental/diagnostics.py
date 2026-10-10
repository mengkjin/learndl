"""Comparable portfolio summaries and plots, independent of PPO training."""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure


def compare_metrics(ppo: dict[str, float], equal: dict[str, float]) -> dict[str, float]:
    """Return differences with explicit units (percentage points versus NAV ratios)."""
    return {
        "return_difference_pp": 100.0 * (ppo["total_return"] - equal["total_return"]),
        "relative_nav_return": ppo["final_nav"] / equal["final_nav"] - 1.0,
        "max_drawdown_difference_pp": 100.0 * (ppo["max_drawdown"] - equal["max_drawdown"]),
        "volatility_difference_pp": 100.0 * (ppo["annualized_volatility"] - equal["annualized_volatility"]),
        "mean_turnover_difference": ppo["mean_turnover"] - equal["mean_turnover"],
        "total_fees_difference": ppo["total_fees"] - equal["total_fees"],
    }


def comparison_figure(histories: dict[str, list[dict[str, Any]]]) -> Figure:
    """All curves start from cash; both decision and settlement dates must match."""
    reference = [(row["decision_date"], row["return_end_date"]) for row in histories["ppo"]]
    if not reference:
        raise ValueError("comparison requires nonempty histories")
    for name, history in histories.items():
        dates = [(row["decision_date"], row["return_end_date"]) for row in history]
        if dates != reference:
            raise ValueError(f"{name} comparison dates do not match PPO")
    figure = Figure(figsize=(13, 12), layout="constrained")
    axes = figure.subplots(3, 2, sharex=True).ravel()
    equal_nav = np.r_[1.0, [row["nav"] for row in histories["equal_weight"]]]
    for name, history in histories.items():
        nav = np.r_[1.0, [row["nav"] for row in history]]
        series = [
            nav,
            nav / np.maximum.accumulate(nav) - 1.0,
            np.r_[0.0, np.cumsum([row["fee_estimate"] for row in history])],
            np.r_[0.0, [row["turnover"] for row in history]],
            np.r_[1.0, [row["cash_weight"] for row in history]],
            nav / equal_nav - 1.0,
        ]
        for ax, values in zip(axes, series, strict=True):
            ax.plot(values, label=name)
    labels = ["NAV", "Drawdown", "Cumulative fees / initial NAV", "Turnover", "Cash weight", "Relative NAV vs equal weight"]
    tick_positions = np.unique(np.linspace(0, len(reference), min(6, len(reference) + 1)).astype(int))
    dates = ["Start"] + [str(date[1]) for date in reference]
    for ax, label in zip(axes, labels, strict=True):
        ax.set_title(label)
        ax.set_xticks(tick_positions, [dates[i] for i in tick_positions], rotation=25)
        ax.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Test period — shared execution, net of trading fees")
    return figure
