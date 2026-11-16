"""Read-only candidate context contract and bounded-memory coverage statistics.

NaN is intentional: these views are NOT registered as training inputs. Validity
is exported separately; a complete-window audit must precede mask-free training.
"""

from __future__ import annotations

from typing import Any
from collections import deque
import hashlib
import json
import numpy as np
import pandas as pd
from .schema import FACTORS, EVENTS

__all__ = [
    'VERSION',
    'RISK_NAMES',
    'PERF_NAMES',
    'MARKET_NAMES',
    'dictionary',
    'candidate_market',
    'context_quality',
    'CoverageState',
    'strata',
    'event_summary',
]

VERSION = "context-audit-2-masked"
RISK_NAMES = tuple(
    "market_day_" + n
    for n in (
        "true_range",
        "turnover",
        "largebuy_price_deviation",
        "smallbuy_percentage",
        "sqrt_avg_size",
        "open_close_percentage",
        "5min_ret_volatility",
        "5min_ret_skewness",
    )
)
PERF_NAMES = tuple(f"{f.lstrip('+-')}__{m}" for f in FACTORS for m in ("ic", "spread"))
MARKET_NAMES = (
    tuple(n for e in EVENTS for n in (e, e + "__age")) + ("market_up",) + RISK_NAMES
)


def dictionary() -> dict[str, Any]:
    features = []
    for name in PERF_NAMES:
        features.append(
            dict(
                block="factorperf",
                name=name,
                units="rank correlation"
                if name.endswith("__ic")
                else "percentage points",
                rule="previous trading day directed raw factor, today realized return; min 50; up to 10 tie-preserving groups; minimum ceil(N/50) per group",
            )
        )
    for name in MARKET_NAMES:
        rule = (
            "log1p(min(trading_day_age,252))/log1p(252)"
            if name.endswith("__age")
            else "past 120 days excluding today; min 60 finite; population std; clip [-5,5]"
            if name in RISK_NAMES
            else "0/1; missing remains unknown"
        )
        features.append(dict(block="factormarket", name=name, rule=rule))
    result = dict(
        version=VERSION,
        factors=list(FACTORS),
        features=features,
        training_enabled=False,
        missing="NaN in audit; paired __valid channels in repaired input; pack zeros only at model boundary",
        masked_input_channels={"factorperf":132,"factormarket":30},
        availability="end-of-day only; historical source revisions not reconstructed",
    )
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    return result


def candidate_market(raw):
    """Input must include prehistory; returns ordered 15-channel view, no filling."""
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Expected unique increasing trading dates")
    out = pd.DataFrame(index=raw.index)
    for event in EVENTS:
        flag = pd.to_numeric(raw[event], errors="coerce")
        if not flag.dropna().isin([0, 1]).all():
            raise ValueError(f"Non-binary event {event}")
        age, last = [], None
        for i, v in enumerate(flag):
            if pd.isna(v):
                # An unavailable day may hide a trigger: forget the previous age.
                last = None
            elif v == 1:
                last = i
            age.append(
                np.nan if last is None else np.log1p(min(i - last, 252)) / np.log1p(252)
            )
        out[event], out[event + "__age"] = flag, age
    ret = pd.to_numeric(raw.market_return, errors="coerce")
    out["market_up"] = (ret > 0).astype(float).where(np.isfinite(ret))
    for name in RISK_NAMES:
        x = pd.to_numeric(raw[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        history = x.shift(1).rolling(120, min_periods=60)
        std = history.std(ddof=0)
        z = (x - history.mean()) / std.where(std > 1e-12)
        # A mature constant reference has no meaningful risk z-score.
        out[name] = z.clip(-5, 5)
    return out.loc[:, list(MARKET_NAMES)]


def context_quality(frame):
    valid = np.isfinite(frame)
    result = pd.DataFrame(index=frame.index)
    result["valid_channels"] = valid.sum(axis=1)
    result["complete"] = valid.all(axis=1)
    result["window_ready"] = np.arange(len(frame)) >= 119
    result["complete_120"] = (
        result.complete.astype(float).rolling(120, min_periods=120).min()
    )
    return result


class CoverageState:
    """Only 60 daily sets retained. Universe changes break sequence completeness."""

    def __init__(self, factors=FACTORS, window=60):
        self.factors, self.window = tuple(factors), window
        self.history = deque(maxlen=window)
        self.last_date = None

    def update(self, date, values, universe):
        if self.last_date is not None and date <= self.last_date:
            raise ValueError("Dates must be strictly increasing, without replay")
        self.last_date = date
        universe = pd.Index(universe)
        if universe.has_duplicates or values.index.has_duplicates:
            raise ValueError("Duplicate security identifiers")
        x = values.reindex(
            index=universe, columns=[f.lstrip("+-") for f in self.factors]
        )
        valid = np.isfinite(x.to_numpy(dtype=float))
        missing = (~valid).sum(axis=1)
        complete = set(universe[missing == 0])
        self.history.append(complete)
        n = len(universe)

        def rate(count):
            return count / n if n else np.nan

        row = dict(
            date=int(date),
            eligible=n,
            complete=len(complete),
            complete_rate=rate(len(complete)),
            window_ready=len(self.history) == self.window,
            complete_60_rate=rate(len(set.intersection(*self.history)))
            if len(self.history) == self.window
            else np.nan,
        )
        for label, mask in [
            ("0", missing == 0),
            ("1", missing == 1),
            ("2_3", (missing >= 2) & (missing <= 3)),
            ("4_10", (missing >= 4) & (missing <= 10)),
            ("gt10", missing > 10),
        ]:
            row["missing_" + label] = int(mask.sum())
        factors = []
        for k, name in enumerate(self.factors):
            factors.append(
                dict(
                    date=int(date),
                    factor=name,
                    valid=int(valid[:, k].sum()),
                    missing=int((~valid[:, k]).sum()),
                    coverage=rate(valid[:, k].sum()),
                    leave_one_out_gain=rate(((missing == 1) & ~valid[:, k]).sum()),
                )
            )
        return row, factors, pd.Series(missing == 0, index=universe, name="complete")


def strata(date, complete, groups, kind):
    groups = groups.reindex(complete.index).fillna("unknown").astype(str)
    return [
        dict(
            date=int(date),
            kind=kind,
            group=g,
            eligible=len(p),
            complete=int(p.sum()),
            retained=float(p.mean()),
        )
        for g, p in complete.groupby(groups)
    ]


def event_summary(raw):
    rows = []
    for event in EVENTS:
        x = raw[event]
        starts = x.eq(1) & ~x.shift().eq(1)
        for year, part in x.groupby(x.index // 10000):
            rows.append(
                dict(
                    event=event,
                    year=int(year),
                    available=int(part.notna().sum()),
                    triggers=int(part.eq(1).sum()),
                    segments=int(starts.loc[part.index].sum()),
                )
            )
    return pd.DataFrame(rows)
