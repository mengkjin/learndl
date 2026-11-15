"""Independent read-only risk references; never writes corrected production data."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

__all__ = [
    'minute_reference',
    'source_quality',
    'RISK_NOTES',
]


def minute_reference(minutes, weights):
    x = minutes.reset_index() if "secid" not in minutes else minutes.copy()
    keys = ["secid"] + (["date"] if "date" in x else [])
    if x.duplicated(keys + ["minute"]).any():
        raise ValueError("Duplicate security/date/minute")
    x = x.sort_values(keys + ["minute"])
    prev = x.groupby(keys, sort=False).close.shift(1)
    x["ret"] = (x.close / prev - 1).replace([np.inf, -np.inf], np.nan)
    x["weight"] = x.secid.map(weights)
    rows = []
    groupkeys = (["date"] if "date" in x else []) + ["minute"]
    for key, part in x.groupby(groupkeys):
        ok = np.isfinite(part.ret) & np.isfinite(part.weight) & (part.weight > 0)
        den = part.weight.where(np.isfinite(part.weight) & (part.weight > 0), 0).sum()
        valid = part.loc[ok, "weight"].sum()
        rows.append(
            dict(
                minute=key[-1] if isinstance(key, tuple) else key,
                date=key[0] if isinstance(key, tuple) and len(key) > 1 else None,
                reference=(part.loc[ok, "ret"] * part.loc[ok, "weight"]).sum() / valid
                if valid
                else np.nan,
                production_denominator_reference=(
                    part.loc[ok, "ret"] * part.loc[ok, "weight"]
                ).sum()
                / den
                if den
                else np.nan,
                valid_weight_coverage=valid / den if den else np.nan,
                invalid_fraction=float((~ok).mean()),
            )
        )
    return pd.DataFrame(rows)


def source_quality(inputs) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Coverage BEFORE zero substitution, using the updater's actual denominators."""
    q, v, mf, minute = (inputs[k] for k in ("quote", "val", "moneyflow", "min"))
    cap = v.float_share * q.preclose
    amount = q.amount
    tr = (
        pd.concat(
            [q.high - q.low, (q.high - q.preclose).abs(), (q.low - q.preclose).abs()],
            axis=1,
        ).max(axis=1)
        / q.preclose
    )
    large = (
        abs(
            (mf.buy_elg_amount + mf.buy_lg_amount)
            / (mf.buy_elg_vol + mf.buy_lg_vol)
            * 100
            - q.vwap
        )
        / q.vwap
    )
    small = mf.buy_sm_amount / q.amount * 10
    oc = (
        minute.query("minute <= 5 or minute >= 42")
        .groupby("secid")
        .amount.sum(min_count=1)
        / q.amount
        / 1000
    )
    rows = []
    for name, x, w in [
        ("true_range", tr, cap),
        ("turnover", q.turn_fl / 100, cap),
        ("largebuy_price_deviation", large, amount),
        ("smallbuy_percentage", small, amount),
        ("open_close_percentage", oc, amount),
    ]:
        x, w = x.reindex(q.index), w.reindex(q.index)
        ok = np.isfinite(x)
        usable = w.where(np.isfinite(w) & (w > 0), 0)
        rows.append(
            dict(
                metric="market_day_" + name,
                invalid_fraction=float((~ok).mean()),
                valid_weight_coverage=usable[ok].sum() / usable.sum()
                if usable.sum()
                else np.nan,
                weight="float_share * preclose"
                if name in ("true_range", "turnover")
                else "daily amount",
                reference=(x[ok] * usable[ok]).sum() / usable[ok].sum()
                if usable[ok].sum()
                else np.nan,
            )
        )
    fallback = "num_trades" not in minute
    if not fallback:
        count = minute.groupby("secid").num_trades.sum(min_count=1)
    else:
        buy = mf[[f"buy_{s}_amount" for s in ("sm", "md", "lg", "elg")]] / [
            5,
            20,
            100,
            500,
        ]
        sell = mf[[f"sell_{s}_amount" for s in ("sm", "md", "lg", "elg")]] / [
            5,
            20,
            100,
            500,
        ]
        count = (buy.sum(axis=1, min_count=4) + sell.sum(axis=1, min_count=4)) / 2 * 10
    count = count.reindex(q.index)
    ok = np.isfinite(count) & (count > 0) & np.isfinite(amount) & (amount >= 0)
    rows.append(
        dict(
            metric="market_day_sqrt_avg_size",
            invalid_fraction=float((~ok).mean()),
            valid_weight_coverage=amount[ok].sum() / amount.sum()
            if amount.sum()
            else np.nan,
            weight="ratio of aggregate amount to aggregate count; not cap weighted",
            reference=np.sqrt(amount[ok].sum() / count[ok].sum())
            if count[ok].sum()
            else np.nan,
            trade_count_fallback=fallback,
        )
    )
    ref = minute_reference(minute, cap)
    # Compare grouped returns with identical production denominator separately
    # from the valid-only denominator, to isolate two different effects.
    r = ref.loc[ref.minute >= 1]
    for name, stat in [("5min_ret_volatility", "std"), ("5min_ret_skewness", "skew")]:
        rows.append(
            dict(
                metric="market_day_" + name,
                invalid_fraction=ref.invalid_fraction.mean(),
                valid_weight_coverage=ref.valid_weight_coverage.mean(),
                weight="minute float_share * preclose",
                reference=getattr(r.reference, stat)(),
                grouped_same_denominator=getattr(
                    r.production_denominator_reference, stat
                )(),
            )
        )
    return rows, ref


RISK_NOTES = {
    "status": "requires_source_quality_review",
    "historical_bug": "v1 used ungrouped close.shift(1). Production now sorts and groups by security/date; historical data require the explicit validated rebuild script. This audit never overwrites source data.",
    "units_to_verify": "large buy amount/volume *100; small buy /daily amount *10; minute amount/daily amount /1000; sqrt amount/count uses source amount units",
    "minute_boundaries": "production selects minute<=5 or minute>=42, and minute>=1 for return moments; verify source numbering",
    "fallback": "trade counts estimated from buy/sell bins [5,20,100,500], averaged then *10; not observed counts",
    "asof": "daily/minute aggregates available after close only; vendor arrival timestamp unavailable unless independently recorded",
    "revision_policy": "independent audit only; any correction/recalculation must create a new source version",
}
