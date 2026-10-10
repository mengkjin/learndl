from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioConstraints:
    candidate_count: int = 100
    max_holdings: int = 50
    max_weight: float = 0.03
    fee_rate: float = 0.00035
    selection_scale: float = 1.0

    @property
    def max_slots(self) -> int:
        return self.candidate_count + self.max_holdings

    def validate(self) -> None:
        if self.candidate_count <= 0 or self.max_holdings <= 0:
            raise ValueError("candidate_count and max_holdings must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")


@dataclass(frozen=True)
class ExecutionResult:
    end_weights: np.ndarray
    end_cash: float
    nav_multiplier: float
    overnight_multiplier: float
    turnover: float
    cost_fraction: float
    buy_notional: float
    sell_notional: float
    frozen_count: int
    rejected_buy_count: int
    rejected_sell_count: int


def select_slots(
    alpha: np.ndarray,
    investable: np.ndarray,
    current_weights: np.ndarray,
    stock_ids: np.ndarray,
    constraints: PortfolioConstraints,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select candidates plus existing positions into fixed-capacity slots."""
    constraints.validate()
    eligible = np.flatnonzero(investable)
    order = np.lexsort((stock_ids[eligible], -alpha[eligible]))
    candidates = eligible[order[: constraints.candidate_count]]
    held = np.flatnonzero(current_weights > 1e-10)
    candidate_set = set(candidates.tolist())
    extras = np.array([i for i in held if i not in candidate_set], dtype=np.int64)
    if extras.size:
        extras = extras[np.argsort(stock_ids[extras], kind="stable")]
    selected = np.concatenate([candidates, extras[: constraints.max_holdings]])
    slots = np.full(constraints.max_slots, -1, dtype=np.int64)
    slots[: len(selected)] = selected
    present = slots >= 0
    can_invest = np.zeros_like(present)
    can_invest[present] = investable[slots[present]]
    return slots, present, can_invest


def _capped_proportional(scores: np.ndarray, total: float, cap: float) -> np.ndarray:
    result = np.zeros_like(scores, dtype=np.float64)
    active = np.ones(scores.size, dtype=bool)
    remaining = min(float(total), cap * scores.size)
    positive = np.maximum(scores.astype(np.float64), 1e-12)
    while remaining > 1e-12 and active.any():
        proposal = remaining * positive[active] / positive[active].sum()
        idx = np.flatnonzero(active)
        saturated = proposal >= cap - result[idx] - 1e-12
        if not saturated.any():
            result[idx] += proposal
            break
        sat_idx = idx[saturated]
        additions = cap - result[sat_idx]
        result[sat_idx] = cap
        remaining -= additions.sum()
        active[sat_idx] = False
    return result.astype(np.float32)


def project_action(
    action: np.ndarray,
    alpha: np.ndarray,
    investable_mask: np.ndarray,
    stock_ids: np.ndarray,
    constraints: PortfolioConstraints,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Map raw selection/allocation preferences to a legal long-only portfolio."""
    constraints.validate()
    action = np.asarray(action, dtype=np.float32).reshape(-1, 2)
    n = len(alpha)
    if action.shape[0] != n or investable_mask.shape != (n,) or stock_ids.shape != (n,):
        raise ValueError("slot arrays do not share the same length")
    valid = np.flatnonzero(investable_mask)
    weights = np.zeros(n, dtype=np.float32)
    if not valid.size:
        return weights, 1.0, np.empty(0, dtype=np.int64)
    rank_score = alpha[valid] + constraints.selection_scale * np.clip(action[valid, 0], -10, 10)
    order = np.lexsort((stock_ids[valid], -rank_score))
    chosen = valid[order[: constraints.max_holdings]]
    allocation_score = np.logaddexp(0.0, np.clip(action[chosen, 1], -20, 20))
    weights[chosen] = _capped_proportional(allocation_score, 1.0, constraints.max_weight)
    cash = float(max(0.0, 1.0 - weights.sum(dtype=np.float64)))
    return weights, cash, chosen


def post_cost_fraction(old_weights: np.ndarray, target_weights: np.ndarray, fee_rate: float) -> float:
    """Solve x = 1 - fee * |x*w_target - w_old| for post-cost NAV fraction."""
    old = np.asarray(old_weights, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    x = 1.0
    for _ in range(100):
        new_x = 1.0 - fee_rate * np.abs(x * target - old).sum()
        if abs(new_x - x) < 1e-12:
            return float(max(new_x, 0.0))
        x = new_x
    raise RuntimeError("transaction-cost fixed point did not converge")


def settle_period(
    old_weights: np.ndarray,
    target_weights: np.ndarray,
    target_cash: float,
    forward_returns: np.ndarray,
    fee_rate: float,
) -> tuple[np.ndarray, float, float, float, float]:
    """Trade and mark one period.

    Returns end weights, end cash weight, NAV multiplier, traded notional and
    transaction cost, with the last two expressed relative to beginning NAV.
    """
    old = np.asarray(old_weights, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    returns = np.nan_to_num(np.asarray(forward_returns, dtype=np.float64), nan=0.0)
    nav_after_cost = post_cost_fraction(old, target, fee_rate)
    traded = np.abs(nav_after_cost * target - old).sum()
    transaction_cost = 1.0 - nav_after_cost
    end_assets = nav_after_cost * target * (1.0 + returns)
    end_cash = nav_after_cost * float(target_cash)
    final_nav = float(end_cash + end_assets.sum())
    if final_nav <= 0 or not np.isfinite(final_nav):
        raise FloatingPointError("portfolio NAV became non-positive or non-finite")
    return (
        (end_assets / final_nav).astype(np.float32),
        end_cash / final_nav,
        final_nav,
        float(traded),
        float(transaction_cost),
    )


def execute_open_close(
    old_weights: np.ndarray,
    old_cash: float,
    target_weights: np.ndarray,
    overnight_returns: np.ndarray,
    intraday_returns: np.ndarray,
    can_buy: np.ndarray,
    can_sell: np.ndarray,
    *,
    fee_rate: float,
    max_holdings: int,
    max_weight: float,
) -> ExecutionResult:
    """Mark close-to-open, execute legal orders, then mark open-to-close.

    All notionals are expressed relative to beginning close NAV. Sells occur
    before buys; unavailable orders remain in the old position. Missing prices
    are accepted only for assets with zero exposure.
    """
    old = np.asarray(old_weights, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    overnight = np.asarray(overnight_returns, dtype=np.float64)
    intraday = np.asarray(intraday_returns, dtype=np.float64)
    buy_mask = np.asarray(can_buy, dtype=bool)
    sell_mask = np.asarray(can_sell, dtype=bool)
    n = old.size
    if any(value.shape != (n,) for value in (target, overnight, intraday, buy_mask, sell_mask)):
        raise ValueError("execution arrays must have identical [stock] shape")
    if np.any(old < -1e-10) or old_cash < -1e-10 or np.any(target < -1e-10):
        raise ValueError("execution supports long-only portfolios")
    if np.any(~np.isfinite(overnight[old > 1e-12])):
        bad = np.flatnonzero((old > 1e-12) & ~np.isfinite(overnight))
        raise ValueError(f"held assets lack an overnight valuation: {bad[:10].tolist()}")

    open_assets = old * (1.0 + np.nan_to_num(overnight, nan=0.0))
    cash = float(old_cash)
    open_nav = float(open_assets.sum() + cash)
    if not np.isfinite(open_nav) or open_nav <= 0:
        raise FloatingPointError("portfolio open NAV became non-positive or non-finite")
    desired = np.minimum(target * open_nav, max_weight * open_nav)

    sell_request = np.maximum(open_assets - desired, 0.0)
    sell_amount = np.where(sell_mask, sell_request, 0.0)
    rejected_sell = (sell_request > 1e-12) & ~sell_mask
    open_assets -= sell_amount
    sell_total = float(sell_amount.sum())
    cash += sell_total * (1.0 - fee_rate)

    buy_request = np.maximum(desired - open_assets, 0.0)
    rejected_buy = (buy_request > 1e-12) & ~buy_mask
    buy_request[~buy_mask] = 0.0

    held = open_assets > 1e-12
    free_slots = max(0, int(max_holdings) - int(held.sum()))
    new_names = np.flatnonzero((~held) & (buy_request > 1e-12))
    if len(new_names) > free_slots:
        keep = new_names[np.argsort(-buy_request[new_names], kind="stable")[:free_slots]]
        dropped = np.setdiff1d(new_names, keep, assume_unique=True)
        rejected_buy[dropped] = True
        buy_request[dropped] = 0.0

    requested_total = float(buy_request.sum())
    affordable = cash / (1.0 + fee_rate) if fee_rate < 1.0 else 0.0
    scale = min(1.0, affordable / requested_total) if requested_total > 0 else 0.0
    buy_amount = buy_request * scale
    buy_total = float(buy_amount.sum())
    cash -= buy_total * (1.0 + fee_rate)
    if cash < -1e-9:
        raise FloatingPointError("buy scaling produced negative cash")
    cash = max(0.0, cash)
    open_assets += buy_amount

    if np.any(~np.isfinite(intraday[open_assets > 1e-12])):
        bad = np.flatnonzero((open_assets > 1e-12) & ~np.isfinite(intraday))
        raise ValueError(f"held assets lack a close valuation: {bad[:10].tolist()}")
    close_assets = open_assets * (1.0 + np.nan_to_num(intraday, nan=0.0))
    final_nav = float(close_assets.sum() + cash)
    if not np.isfinite(final_nav) or final_nav <= 0:
        raise FloatingPointError("portfolio close NAV became non-positive or non-finite")
    costs = fee_rate * (buy_total + sell_total)
    frozen = (open_assets > 1e-12) & (~buy_mask | ~sell_mask)
    return ExecutionResult(
        end_weights=(close_assets / final_nav).astype(np.float32),
        end_cash=float(cash / final_nav),
        nav_multiplier=final_nav,
        overnight_multiplier=open_nav,
        turnover=float(buy_total + sell_total),
        cost_fraction=float(costs),
        buy_notional=buy_total,
        sell_notional=sell_total,
        frozen_count=int(frozen.sum()),
        rejected_buy_count=int(rejected_buy.sum()),
        rejected_sell_count=int(rejected_sell.sum()),
    )
