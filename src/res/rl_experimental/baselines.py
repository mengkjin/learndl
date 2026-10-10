from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .data import PanelData
from .portfolio import PortfolioConstraints, execute_open_close, settle_period


BaselineKind = Literal["alpha_equal_weight", "robust_top50"]


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        for row in history:
            writer.writerow({
                key: json.dumps(value, default=_json_default) if isinstance(value, (list, np.ndarray)) else value
                for key, value in row.items()
            })


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, (int, float)):
            return float(scalar)
    raise TypeError(f"expected a number, got {type(value).__name__}")


def history_metrics(history: list[dict[str, object]]) -> dict[str, float]:
    if not history:
        return {"total_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0, "mean_turnover": 0.0}
    nav = np.asarray([_as_float(row["nav"]) for row in history])
    returns = np.asarray([_as_float(row["period_return"]) for row in history])
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))
    drawdown = np.concatenate([[1.0], nav]) / peak - 1.0
    metrics = {
        "total_return": float(nav[-1] - 1.0),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(np.mean([_as_float(row["turnover"]) for row in history])),
        "final_nav": float(nav[-1]),
        "total_fees": float(np.sum([_as_float(row.get("fee_estimate", 0.0)) for row in history])),
    }
    if "frozen_holdings" in history[0]:
        metrics["mean_frozen_holdings"] = float(np.mean([_as_float(row["frozen_holdings"]) for row in history]))
        metrics["rejected_orders"] = float(
            np.sum([_as_float(row["rejected_buys"]) + _as_float(row["rejected_sells"]) for row in history])
        )
    return metrics


def _equal_weight_target(panel: PanelData, time_index: int, constraints: PortfolioConstraints) -> np.ndarray:
    eligible = np.flatnonzero(panel.investable[time_index])
    order = np.lexsort((panel.stock_ids[eligible], -panel.alpha[time_index, eligible]))
    chosen = eligible[order[: constraints.max_holdings]]
    target = np.zeros(panel.n_stocks, dtype=np.float32)
    if chosen.size:
        allocation = min(constraints.max_weight, 1.0 / chosen.size)
        target[chosen] = allocation
    return target


def _robust_target(
    panel: PanelData,
    time_index: int,
    current_weights: np.ndarray,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Dependency-free translation of the project's robust top-stock rule."""
    eligible = np.flatnonzero(panel.investable[time_index])
    target = np.zeros(panel.n_stocks, dtype=np.float32)
    if eligible.size == 0:
        return target
    alpha = panel.alpha[time_index, eligible]
    ids = panel.stock_ids[eligible]
    industry = panel.industry_codes[time_index, eligible] if panel.industry_codes is not None else np.full(len(eligible), -1)
    descending = np.lexsort((ids, -alpha))
    ascending = np.lexsort((ids, alpha))
    rank_percentile = np.empty(len(eligible), dtype=float)
    rank_percentile[ascending] = (np.arange(len(eligible)) + 1) / len(eligible)
    industry_rank = np.empty(len(eligible), dtype=float)
    for code in np.unique(industry):
        members = np.flatnonzero(industry == code)
        order = members[np.lexsort((ids[members], -alpha[members]))]
        industry_rank[order] = np.arange(1, len(order) + 1)

    selected = current_weights[eligible] > 1e-10
    selected_cumulative = np.empty(len(eligible), dtype=int)
    selected_cumulative[descending] = np.cumsum(selected[descending])
    stay_number = (1.0 - 0.1) * constraints.max_holdings
    buffered = (rank_percentile >= 0.8) | ((selected_cumulative <= stay_number) & (rank_percentile >= 0.5))
    stay = selected & buffered
    stay_count = {int(code): int(np.count_nonzero(stay & (industry == code))) for code in np.unique(industry)}
    industry_slots = constraints.max_holdings * 0.1
    entry_ok = np.asarray([
        (not stay[index]) and industry_rank[index] < industry_slots - stay_count[int(industry[index])]
        for index in range(len(eligible))
    ])
    entry_order = [index for index in descending if entry_ok[index]][: max(0, constraints.max_holdings - int(stay.sum()))]
    chosen_local = np.union1d(np.flatnonzero(stay), np.asarray(entry_order, dtype=np.int64))
    if chosen_local.size:
        target[eligible[chosen_local]] = min(constraints.max_weight, 1.0 / chosen_local.size)
    return target


def project_robust_target(
    panel: PanelData,
    time_index: int,
    current_weights: np.ndarray,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Reference wrapper used to check the translation against learndl."""
    from src.res.factor.fmp.generator.top import TopStocksPortfolioCreator
    from src.res.factor.util import Amodel, Port

    date = int(panel.decision_dates[time_index])
    eligible = panel.investable[time_index]
    alpha = Amodel(date, panel.alpha[time_index, eligible].astype(float), panel.stock_ids[eligible], "rl_snapshot_alpha")
    held = current_weights > 1e-10
    initial = Port.create(panel.stock_ids[held], current_weights[held], date=date, name="rl_actual_holdings")
    creator = TopStocksPortfolioCreator(
        "rl_robust_top50",
        n_best=constraints.max_holdings,
        turn_control=0.1,
        buffer_zone=0.8,
        no_zone=0.5,
        indus_control=0.1,
        vb_level="never",
    )
    result = creator.create(date, alpha, benchmark=None, init_port=initial).port
    target = np.zeros(panel.n_stocks, dtype=np.float32)
    if result and not result.port.empty:
        positions = np.searchsorted(panel.stock_ids, result.secid)
        valid = (positions < panel.n_stocks) & (panel.stock_ids[np.minimum(positions, panel.n_stocks - 1)] == result.secid)
        target[positions[valid]] = np.minimum(result.weight[valid], constraints.max_weight)
    return target


def replay_baseline(
    panel: PanelData,
    interval: tuple[int, int],
    constraints: PortfolioConstraints,
    kind: BaselineKind,
) -> list[dict[str, object]]:
    """Replay a baseline through the same accounting kernel as the RL agent."""
    start, end = interval
    weights = np.zeros(panel.n_stocks, dtype=np.float32)
    cash = 1.0
    nav = 1.0
    history: list[dict[str, object]] = []
    execution = panel.execution_arrays() if panel.is_execution_panel else None
    for time_index in range(start, end):
        if kind == "alpha_equal_weight":
            target = _equal_weight_target(panel, time_index, constraints)
        elif kind == "robust_top50":
            target = _robust_target(panel, time_index, weights, constraints)
        else:
            raise ValueError(f"unknown baseline: {kind}")

        previous_nav = nav
        if execution is not None:
            execution_dates, overnight, intraday, can_buy, can_sell, valuation_stale = execution
            filled = execute_open_close(
                weights,
                cash,
                target,
                overnight[time_index],
                intraday[time_index],
                can_buy[time_index],
                can_sell[time_index],
                fee_rate=constraints.fee_rate,
                max_holdings=constraints.max_holdings,
                max_weight=constraints.max_weight,
            )
            weights, cash = filled.end_weights, filled.end_cash
            multiplier, turnover, cost = filled.nav_multiplier, filled.turnover, filled.cost_fraction
            extra = {
                "execution_date": int(execution_dates[time_index]),
                "overnight_return": filled.overnight_multiplier - 1.0,
                "buy_notional": filled.buy_notional,
                "sell_notional": filled.sell_notional,
                "frozen_holdings": filled.frozen_count,
                "rejected_buys": filled.rejected_buy_count,
                "rejected_sells": filled.rejected_sell_count,
                "stale_valuations": int(np.count_nonzero(valuation_stale[time_index] & (weights > 1e-8))),
            }
        else:
            target_cash = max(0.0, 1.0 - float(target.sum()))
            weights, cash, multiplier, turnover, cost = settle_period(
                weights, target, target_cash, panel.forward_returns[time_index], constraints.fee_rate
            )
            extra = {}
        nav *= multiplier
        held = weights > 1e-8
        history.append(
            {
                "time_index": time_index,
                "decision_date": int(panel.decision_dates[time_index]),
                "return_end_date": int(panel.return_end_dates[time_index]),
                "nav": nav,
                "period_return": multiplier - 1.0,
                "turnover": turnover,
                "fee_estimate": previous_nav * cost,
                "cash_weight": cash,
                "holdings": int(held.sum()),
                "held_stock_ids": panel.stock_ids[held].copy(),
                "held_weights": weights[held].copy(),
            }
            | extra
        )
    return history
