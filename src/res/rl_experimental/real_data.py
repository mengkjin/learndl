from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import CALENDAR, DB

from .data import PANEL_SCHEMA_VERSION, PanelData, date_split
from .data_quality import deduplicate_table
from .progress import progress


STYLE_FEATURES = ("size", "beta", "momentum", "residual_volatility", "liquidity")
MARKET_FEATURES = ("ret_1d", "ret_5d", "ret_20d", "volatility_20d", "turnover_20d", "amount_20d")
CONTINUOUS_FEATURES = MARKET_FEATURES + STYLE_FEATURES


@dataclass(frozen=True)
class RealDataConfig:
    alpha: str
    start: int
    end: int
    output: Path
    alpha_direction: int = 1
    alpha_lag: int = 0
    max_alpha_staleness: int = 0
    min_listing_days: int = 63
    alpha_sample_status: str = "unknown"

    def validate(self) -> None:
        if not self.alpha or "@" not in self.alpha:
            raise ValueError("alpha must use source@name or source@name@column")
        if self.start >= self.end:
            raise ValueError("start must precede end")
        if self.alpha_direction not in (-1, 1):
            raise ValueError("alpha_direction must be -1 or 1")
        if self.alpha_lag < 0 or self.max_alpha_staleness < 0 or self.min_listing_days < 0:
            raise ValueError("lag, staleness and listing days cannot be negative")
        if self.alpha_sample_status not in {"unknown", "out-of-sample", "in-sample"}:
            raise ValueError("alpha_sample_status must be unknown, out-of-sample, or in-sample")


def _alpha_source(expression: str) -> tuple[str, str, str]:
    parts = expression.split("@")
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError("alpha must use source@name or source@name@column")
    source, key = parts[:2]
    source = {"factor": "stock_factor", "pred": "model_prediction"}.get(source, source)
    if source not in {"stock_factor", "model_prediction", "sellside"}:
        raise ValueError("alpha source must be factor, pred, sellside, stock_factor, or model_prediction")
    return source, key, parts[2] if len(parts) == 3 else key


def _calendar_dates(start: int, end: int) -> np.ndarray:
    values = np.asarray(CALENDAR.range(int(start), int(end), "td", until_today=False), dtype=np.int64)
    if values.size == 0:
        raise ValueError(f"no trading dates in [{start}, {end}]")
    return values


def _alpha_date_map(config: RealDataConfig, decision_dates: np.ndarray) -> tuple[dict[int, int], list[int]]:
    source, key, _ = _alpha_source(config.alpha)
    available = np.asarray(DB.dates(source, key), dtype=np.int64)
    if available.size == 0:
        raise FileNotFoundError(f"alpha source {source}/{key} has no stored dates")
    full_calendar = _calendar_dates(int(min(available.min(), decision_dates.min())), int(decision_dates.max()))
    calendar_position = {int(date): i for i, date in enumerate(full_calendar)}
    mapping: dict[int, int] = {}
    missing: list[int] = []
    for decision in decision_dates:
        target = int(CALENDAR.td(int(decision), -config.alpha_lag).as_int())
        pos = int(np.searchsorted(available, target, side="right") - 1)
        if pos < 0:
            missing.append(int(decision))
            continue
        source_date = int(available[pos])
        stale = calendar_position.get(target, -1) - calendar_position.get(source_date, -10**9)
        if stale < 0 or stale > config.max_alpha_staleness:
            missing.append(int(decision))
            continue
        mapping[int(decision)] = source_date
    return mapping, missing


def _load_table(source: str, key: str, dates: np.ndarray, date_key: str,
                audit: list[dict]) -> pd.DataFrame:
    frames = []
    for start in range(0, len(dates), 128):
        batch = dates[start:start + 128]
        progress('data', f'Reading {source}/{key}: dates {start + 1}-{start + len(batch)}/{len(dates)}')
        frames.append(DB.loads(source, key, batch, key_column=date_key,
                               override_existing_key=True, vb_level="never"))
    frame = pd.concat(frames, ignore_index=True)
    if not frame.empty:
        frame = deduplicate_table(frame, [date_key, 'secid'], f'{source}/{key}', audit)
    progress('data', f'Loaded {source}/{key}: {len(frame):,} rows')
    return frame


def _load_alpha(config: RealDataConfig, decision_dates: np.ndarray,
                duplicate_audit: list[dict] | None = None) -> tuple[pd.DataFrame, dict[int, int]]:
    source, key, column = _alpha_source(config.alpha)
    mapping, missing = _alpha_date_map(config, decision_dates)
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"alpha is unavailable under the configured lag/staleness for {len(missing)} dates: {preview}")
    source_dates = np.unique(list(mapping.values()))
    raw = _load_table(source, key, source_dates, "source_date", duplicate_audit if duplicate_audit is not None else [])
    required = {"source_date", "secid", column}
    if not required.issubset(raw.columns):
        raise ValueError(f"alpha {source}/{key} must contain {sorted(required)}; got {raw.columns.tolist()}")
    raw = raw.loc[:, ["source_date", "secid", column]].rename(columns={column: "raw_alpha"})
    reverse: dict[int, list[int]] = {}
    for decision, source_date in mapping.items():
        reverse.setdefault(source_date, []).append(decision)
    frames = []
    for source_date, decisions in reverse.items():
        one = raw.loc[raw.source_date == source_date, ["secid", "raw_alpha"]]
        for decision in decisions:
            frame = one.copy()
            frame["date"] = decision
            frames.append(frame)
    alpha = pd.concat(frames, ignore_index=True)
    alpha["raw_alpha"] = pd.to_numeric(alpha["raw_alpha"], errors="coerce") * config.alpha_direction
    alpha["alpha"] = alpha.groupby("date", sort=False)["raw_alpha"].rank(method="first", pct=True)
    return alpha.loc[:, ["date", "secid", "alpha"]], mapping


def _assert_daily_table(frame: pd.DataFrame, dates: np.ndarray, name: str) -> None:
    if frame.empty:
        raise FileNotFoundError(f"{name} is empty")
    if frame.duplicated(["date", "secid"]).any():
        raise ValueError(f"{name} contains duplicate date/secid rows")
    present = set(frame.date.astype(int).unique())
    missing = [int(date) for date in dates if int(date) not in present]
    if missing:
        raise FileNotFoundError(f"{name} is missing {len(missing)} complete dates: {missing[:10]}")


def _assert_positive_prices(frame: pd.DataFrame, fields: tuple[str, ...], name: str) -> None:
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        invalid = values.notna() & (~np.isfinite(values) | (values <= 0))
        if invalid.any():
            sample = frame.loc[invalid, ["date", "secid", field]].head(10).to_dict("records")
            raise ValueError(f"{name}.{field} contains non-positive/non-finite prices: {sample}")


def _pivot(frame: pd.DataFrame, field: str, dates: np.ndarray, stocks: np.ndarray) -> pd.DataFrame:
    result = frame.pivot(index="date", columns="secid", values=field)
    return result.reindex(index=dates, columns=stocks)


def _estimate_bytes(n_dates: int, n_stocks: int) -> int:
    float_arrays = 1 + 1 + 1 + len(CONTINUOUS_FEATURES) * 2
    bool_arrays = 5
    return n_dates * n_stocks * (float_arrays * 4 + bool_arrays) + n_dates * n_stocks * 2


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_real_data(config: RealDataConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Audit project data and optionally export a versioned real-data panel."""
    config.validate()
    progress('data', f'Preparing {config.alpha}, {config.start}-{config.end}, dry_run={dry_run}')
    duplicate_audit: list[dict] = []
    decision_dates = _calendar_dates(config.start, config.end)
    execution_dates = np.asarray(CALENDAR.offset(decision_dates, 1, "td"), dtype=np.int64)
    alpha_long, alpha_mapping = _load_alpha(config, decision_dates, duplicate_audit)
    description = DATAVENDOR.INFO.get_desc(set_index=False, listed=True, exchange=["SZSE", "SSE"])
    description = deduplicate_table(description, ['secid', 'list_dt', 'delist_dt'],
                                    'information_ts/description', duplicate_audit)
    relevant_description = description[(description.list_dt <= config.end) & (description.delist_dt > config.start)]
    allowed_ids = set(relevant_description.secid.astype(int))
    alpha_long = alpha_long[alpha_long.secid.isin(allowed_ids)]
    stock_ids = np.sort(alpha_long.secid.astype(np.int64).unique())
    if stock_ids.size == 0:
        raise ValueError("alpha has no historical SSE/SZSE stocks in the requested interval")

    source, key, column = _alpha_source(config.alpha)
    audit: dict[str, Any] = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "requested_dates": [int(decision_dates[0]), int(decision_dates[-1])],
        "decision_count": int(len(decision_dates)),
        "stock_count": int(len(stock_ids)),
        "estimated_bytes": _estimate_bytes(len(decision_dates), len(stock_ids)),
        "duplicate_cleanup": duplicate_audit,
        "alpha": {
            "expression": config.alpha,
            "database_source": source,
            "database_key": key,
            "column": column,
            "direction": config.alpha_direction,
            "lag": config.alpha_lag,
            "max_staleness": config.max_alpha_staleness,
            "sample_status": config.alpha_sample_status,
            "source_date_range": [min(alpha_mapping.values()), max(alpha_mapping.values())],
        },
    }
    if dry_run:
        return {"dry_run": True, "audit": audit}

    history_start = int(CALENDAR.td(int(decision_dates[0]), -25).as_int())
    history_dates = _calendar_dates(history_start, int(execution_dates[-1]))
    day = _load_table("trade_ts", "day", history_dates, "date", duplicate_audit)
    adjusted = _load_table("trade_ts", "adjprice", history_dates, "date", duplicate_audit)
    limits = _load_table("trade_ts", "day_limit", execution_dates, "date", duplicate_audit)
    exposure = _load_table("models", "tushare_cne5_exp", decision_dates, "date", duplicate_audit)
    progress('data', 'Checking daily coverage and prices')
    _assert_daily_table(day, history_dates, "trade_ts/day")
    _assert_daily_table(adjusted, history_dates, "trade_ts/adjprice")
    _assert_daily_table(limits, execution_dates, "trade_ts/day_limit")
    _assert_daily_table(exposure, decision_dates, "models/tushare_cne5_exp")
    # Source files also contain securities outside this historical SSE/SZSE snapshot.
    # Their prices must not block export; all snapshot securities remain validated.
    _assert_positive_prices(day[day.secid.isin(stock_ids)], ("open", "close"), "trade_ts/day")
    _assert_positive_prices(adjusted[adjusted.secid.isin(stock_ids)], ("open", "close"), "trade_ts/adjprice")
    _assert_positive_prices(limits[limits.secid.isin(stock_ids)], ("up_limit", "down_limit"), "trade_ts/day_limit")

    progress('data', f'Building features: {len(decision_dates)} decision dates, {len(stock_ids)} stocks')
    close = _pivot(adjusted, "close", history_dates, stock_ids)
    daily_return = close.pct_change(fill_method=None)
    day_indexed = day.set_index(["date", "secid"]).sort_index()
    turnover = day_indexed["turn_tt"].unstack("secid").reindex(index=history_dates, columns=stock_ids)
    amount = day_indexed["amount"].unstack("secid").reindex(index=history_dates, columns=stock_ids)
    mean_amount = amount.rolling(20, min_periods=10).mean()
    feature_frames: dict[str, pd.DataFrame] = {
        "ret_1d": close.pct_change(1, fill_method=None),
        "ret_5d": close.pct_change(5, fill_method=None),
        "ret_20d": close.pct_change(20, fill_method=None),
        "volatility_20d": daily_return.rolling(20, min_periods=15).std(),
        "turnover_20d": turnover.rolling(20, min_periods=10).mean(),
        "amount_20d": pd.DataFrame(
            np.log1p(mean_amount.to_numpy(dtype=np.float64)),
            index=mean_amount.index,
            columns=mean_amount.columns,
        ),
    }
    for name in STYLE_FEATURES:
        feature_frames[name] = _pivot(exposure, name, decision_dates, stock_ids)

    continuous = np.empty((len(decision_dates), len(stock_ids), len(CONTINUOUS_FEATURES)), dtype=np.float32)
    missing = np.empty_like(continuous)
    missing_counts: dict[str, int] = {}
    for feature_index, name in enumerate(CONTINUOUS_FEATURES):
        progress('data', f'Feature {feature_index + 1}/{len(CONTINUOUS_FEATURES)}: {name}')
        values = feature_frames[name].reindex(index=decision_dates, columns=stock_ids).to_numpy(dtype=np.float64)
        missing_mask = ~np.isfinite(values)
        missing_counts[name] = int(missing_mask.sum())
        for row in range(len(values)):
            finite = np.isfinite(values[row])
            fill = float(np.median(values[row, finite])) if finite.any() else 0.0
            values[row, ~finite] = fill
        continuous[:, :, feature_index] = values.astype(np.float32)
        missing[:, :, feature_index] = missing_mask.astype(np.float32)
    features = np.concatenate([continuous, missing], axis=2)
    feature_names = CONTINUOUS_FEATURES + tuple(f"{name}_missing" for name in CONTINUOUS_FEATURES)

    alpha_wide = alpha_long.pivot(index="date", columns="secid", values="alpha").reindex(index=decision_dates, columns=stock_ids)
    alpha_available = np.isfinite(alpha_wide.to_numpy())
    alpha_values = alpha_wide.fillna(0.0).to_numpy(dtype=np.float32)

    listed = np.zeros((len(decision_dates), len(stock_ids)), dtype=bool)
    stock_position = {int(secid): index for index, secid in enumerate(stock_ids)}
    listing = relevant_description.loc[:, ["secid", "list_dt", "delist_dt"]]
    listing_ids = listing.secid.to_numpy(dtype=np.int64)
    listing_starts = listing.list_dt.to_numpy(dtype=np.int64)
    listing_ends = listing.delist_dt.to_numpy(dtype=np.int64)
    for secid, list_dt, delist_dt in zip(listing_ids, listing_starts, listing_ends, strict=True):
        position = stock_position.get(int(secid))
        if position is None:
            continue
        mature_date = int(CALENDAR.td(int(list_dt), config.min_listing_days).as_int())
        listed[:, position] |= (decision_dates >= mature_date) & (decision_dates < int(delist_dt))
    st_mask = np.zeros_like(listed)
    for row, date in enumerate(decision_dates):
        if row % 128 == 0:
            progress('data', f'Listing/ST eligibility: {row + 1}/{len(decision_dates)} dates')
        st_ids = DATAVENDOR.INFO.get_st(int(date)).secid.to_numpy(dtype=np.int64)
        st_mask[row] = np.isin(stock_ids, st_ids)
    day_close = _pivot(day, "close", history_dates, stock_ids).reindex(index=decision_dates)
    traded_at_decision = np.isfinite(day_close.to_numpy()) & (day_close.to_numpy() > 0)
    investable = listed & ~st_mask & alpha_available & traded_at_decision
    # Re-rank after all point-in-time eligibility filters. The earlier rank is
    # monotonic and only supplies a stable tie order across reused source rows.
    for row in range(len(decision_dates)):
        valid = investable[row]
        alpha_values[row, ~valid] = 0.0
        if valid.any():
            alpha_values[row, valid] = pd.Series(alpha_values[row, valid]).rank(method="first", pct=True).to_numpy(np.float32)

    industry_codes = np.full((len(decision_dates), len(stock_ids)), -1, dtype=np.int16)
    raw_industries: list[np.ndarray] = []
    all_industry_values: set[Any] = set()
    for row, date in enumerate(decision_dates):
        if row % 128 == 0:
            progress('data', f'Industry classifications: {row + 1}/{len(decision_dates)} dates')
        industry = DATAVENDOR.INFO.get_indus(int(date))
        industry = deduplicate_table(industry.reset_index(), ['secid'],
                                      f'information_ts/industry/{date}', duplicate_audit).set_index('secid')
        values = industry.reindex(stock_ids).indus.to_numpy()
        raw_industries.append(values)
        all_industry_values.update(value for value in values if pd.notna(value))
    industry_map = {value: index for index, value in enumerate(sorted(all_industry_values, key=str))}
    for row, values in enumerate(raw_industries):
        industry_codes[row] = np.asarray([industry_map.get(value, -1) if pd.notna(value) else -1 for value in values], dtype=np.int16)

    progress('data', 'Building next-open execution masks and overnight/intraday returns')
    close_t = close.reindex(index=decision_dates).to_numpy(dtype=np.float64)
    close_next = close.reindex(index=execution_dates).to_numpy(dtype=np.float64)
    execution_day = day[day.date.isin(execution_dates)].copy()
    raw_open = _pivot(execution_day, "open", execution_dates, stock_ids).to_numpy(dtype=np.float64)
    adj_factor = _pivot(execution_day, "adjfactor", execution_dates, stock_ids).to_numpy(dtype=np.float64)
    status = _pivot(execution_day, "status", execution_dates, stock_ids).to_numpy(dtype=np.float64)
    adjusted_open = raw_open * adj_factor
    limit_up = _pivot(limits, "up_limit", execution_dates, stock_ids).to_numpy(dtype=np.float64)
    limit_down = _pivot(limits, "down_limit", execution_dates, stock_ids).to_numpy(dtype=np.float64)
    valid_open = np.isfinite(adjusted_open) & (adjusted_open > 0) & (status > 0)
    valid_limits = np.isfinite(limit_up) & np.isfinite(limit_down)
    can_buy = valid_open & valid_limits & (raw_open < limit_up - 1e-8)
    can_sell = valid_open & valid_limits & (raw_open > limit_down + 1e-8)
    stale = ~valid_open

    overnight = np.full_like(close_t, np.nan, dtype=np.float64)
    intraday = np.full_like(close_t, np.nan, dtype=np.float64)
    active = valid_open & np.isfinite(close_t) & (close_t > 0) & np.isfinite(close_next) & (close_next > 0)
    overnight[active] = adjusted_open[active] / close_t[active] - 1.0
    intraday[active] = close_next[active] / adjusted_open[active] - 1.0
    carried = stale & np.isfinite(close_t) & np.isfinite(close_next) & np.isclose(close_t, close_next, rtol=1e-6, atol=1e-8)
    overnight[carried] = 0.0
    intraday[carried] = 0.0
    forward = close_next / close_t - 1.0

    panel = PanelData(
        stock_ids=stock_ids,
        decision_dates=decision_dates,
        execution_dates=execution_dates,
        return_end_dates=execution_dates,
        features=features,
        alpha=alpha_values,
        investable=investable,
        forward_returns=forward.astype(np.float32),
        overnight_returns=overnight.astype(np.float32),
        intraday_returns=intraday.astype(np.float32),
        can_buy=can_buy,
        can_sell=can_sell,
        valuation_stale=stale,
        industry_codes=industry_codes,
        feature_names=feature_names,
    )

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "panel.npz"
    progress('data', f'Writing snapshot: {panel_path}')
    panel.save_npz(panel_path)
    quality = {
        "duplicate_cleanup": duplicate_audit,
        "investable_per_day": {
            "min": int(investable.sum(axis=1).min()),
            "median": float(np.median(investable.sum(axis=1))),
            "max": int(investable.sum(axis=1).max()),
        },
        "alpha_coverage": float(alpha_available.sum() / listed.sum()),
        "feature_missing_cells": missing_counts,
        "stale_valuation_cells": int(stale.sum()),
        "cannot_buy_cells": int((investable & ~can_buy).sum()),
        "cannot_sell_cells": int((investable & ~can_sell).sum()),
        "industry_missing_cells": int((industry_codes < 0).sum()),
    }
    default_splits = date_split(panel)
    split_dates = {
        name: {
            "indices": [start, end],
            "decision_start": int(panel.decision_dates[start]),
            "return_end": int(panel.return_end_dates[end - 1]),
        }
        for name, (start, end) in default_splits.items()
    }
    manifest = {
        "config": {**asdict(config), "output": str(output)},
        "audit": audit,
        "panel_contract": panel.contract(),
        "feature_names": list(feature_names),
        "industry_mapping": {str(key): value for key, value in industry_map.items()},
        "calendar": {"decision": "SSE/SZSE trading day", "execution": "next trading-day open", "valuation": "execution-day close"},
        "default_time_splits": split_dates,
        "execution_assumptions": [
            "daily open execution; sells precede buys",
            "suspension/no open freezes the position",
            "open at upper limit cannot buy; open at lower limit cannot sell",
            "no capacity, queue, partial-fill, or market-impact model",
        ],
        "research_status": "engineering experiment" if config.alpha_sample_status == "unknown" else config.alpha_sample_status,
        "panel_sha256": _file_sha256(panel_path),
    }
    (output / "quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    progress('data', f'Snapshot complete: {panel_path}; duplicate rows removed='
             f'{sum(event["removed_rows"] for event in duplicate_audit)}')
    return {"dry_run": False, "panel": str(panel_path), "manifest": manifest, "quality": quality}
