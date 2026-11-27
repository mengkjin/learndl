"""
Market-level aggregation of daily microstructure risk features.

Aggregates the same 8 risk features from ``daily_risk.py`` across all stocks
using feature-specific cap/amount weights and aggregate ratios, producing a market-level
time series stored as a single table in ``market_daily/risk``.

Unlike ``DailyRiskUpdater``, results are appended to one database file rather
than stored per date, using ``append_result`` to merge and deduplicate.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any

from src.proj import CALENDAR, DB, Load, Base, Dates
from src.proj.db.io.output_lock import output_lock
from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ["MarketDailyRiskUpdater"]


class MarketDailyRiskUpdater(BasicCustomUpdater):
    """Registered updater for market-level daily risk features."""

    START_DATE = max(20100101, DB.min_date("trade_ts", "5min", use_alt=True))
    DB_SRC = "market_daily"
    DB_KEY = "risk"

    @classmethod
    def calculate(cls, date: int, *, backend: str = "polars", inputs=None):
        """Compute one date; Polars independently replaces four minute-derived metrics."""
        return calc_market_daily_risk(date, inputs=inputs, backend=backend)

    @classmethod
    def proceed_update(
        cls,
        start: int | None = None,
        end: int | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> Base.UpdateFlag:
        """Update market-level risk features for all missing dates and append to the data."""
        start = max(start or cls.START_DATE, cls.START_DATE)
        end = end or min(
            CALENDAR.updated(), DB.max_date("trade_ts", "5min", use_alt=True)
        )
        if overwrite:
            stored_dates = Dates()
        else:
            stored_df = Load.df(DB.path(cls.DB_SRC, cls.DB_KEY))
            stored_dates = (
                Dates()
                if stored_df.empty
                else Dates(stored_df.reset_index()["date"].to_numpy(int))
            )
        target_dates = Dates(start, end).diff(stored_dates)

        if target_dates.empty:
            return Base.UpdateFlag.SKIPPED

        new_dfs: list[pd.DataFrame] = []
        for date in target_dates:
            new_dfs.append(calc_market_daily_risk(date))
            cls.logger.stdout(f"Calculate market daily risk at {date}", idt=2, vb=2)

        cls.append_result(pd.concat(new_dfs))
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls, date: int):
        """Compute and append market risk features for a single ``date``."""
        cls.append_result(calc_market_daily_risk(date))

    @classmethod
    def append_result(cls, new_df: pd.DataFrame, *, save=None):
        """
        Merge ``new_df`` with the existing database table, deduplicate by date, and overwrite.

        Keeps the ``last`` value when duplicate dates exist (allows reprocessing).
        """
        with output_lock(DB.path(cls.DB_SRC, cls.DB_KEY)):
            previous = DB.load(cls.DB_SRC, cls.DB_KEY)
            combined = pd.concat([previous, new_df])
            if combined.empty:
                return
            combined = combined.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
            if save is None:
                DB.save(combined, cls.DB_SRC, cls.DB_KEY, indent=cls.logger.indent + 1,
                        vb_level=cls.logger.vb_level + 1)
            else:
                save(combined, cls.DB_SRC, cls.DB_KEY)


def _get_inputs(date: int) -> dict:
    inputs = {
        "quote": DB.load("trade_ts", "day", date),
        "val": DB.load("trade_ts", "day_val", date),
        "moneyflow": DB.load("trade_ts", "day_moneyflow", date),
    }
    for name, df in inputs.items():
        if not df.empty:
            inputs[name] = df.set_index("secid")
    return {**inputs, "min": DB.load_pl("trade_ts", "5min", date, use_alt=True)}


def _fillinf(series: pd.Series, fill_value: float | Any = 0) -> pd.Series:
    return series.where(np.isfinite(series), fill_value)


def calc_market_daily_risk(date: int, *, inputs=None, backend="polars"):
    if backend not in ("pandas", "polars"):
        raise ValueError(f"Unknown risk backend: {backend}")
    inputs = _get_inputs(date) if inputs is None else inputs
    if backend == "pandas" and not isinstance(inputs["min"], pd.DataFrame):
        inputs = {**inputs, "min": inputs["min"].to_pandas().set_index("secid")}
    funcs = [
        market_day_true_range,
        market_day_turnover,
        market_day_largebuy_price_deviation,
        market_day_smallbuy_percentage,
        market_day_sqrt_avg_size,
        market_day_open_close_percentage,
        market_day_5min_ret_volatility,
        market_day_5min_ret_skewness,
    ]
    metrics = {
        func.__name__: func(**inputs)
        for func in (funcs if backend == "pandas" else funcs[:4])
    }
    if backend == "polars":
        metrics.update(
            polars_minutes(
                inputs["quote"], inputs["val"], inputs["min"], inputs["moneyflow"]
            )[0]
        )
    result = pd.DataFrame(metrics, index=pd.Index([date], name="date")).reset_index()
    return result


def market_day_true_range(quote: pd.DataFrame, val: pd.DataFrame, **kwargs) -> float:
    tr = pd.concat(
        [
            quote["high"] - quote["low"],
            (quote["high"] - quote["preclose"]).abs(),
            (quote["low"] - quote["preclose"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = _fillinf((tr / quote["preclose"]).rename("true_range"), 0)
    weight = (val["float_share"] * quote["preclose"]).fillna(0)
    return tr.fillna(0).mul(weight).sum() / weight.sum()


def market_day_turnover(quote: pd.DataFrame, val: pd.DataFrame, **kwargs) -> float:
    turnover = quote["turn_fl"] / 100
    weight = (val["float_share"] * quote["preclose"]).fillna(0)
    return turnover.mul(weight).sum() / weight.sum()


def market_day_largebuy_price_deviation(
    quote: pd.DataFrame, moneyflow: pd.DataFrame, **kwargs
) -> float:
    q = quote.join(
        moneyflow.loc[
            :, ["buy_elg_amount", "buy_elg_vol", "buy_lg_amount", "buy_lg_vol"]
        ]
    )
    q["lbp"] = (
        (q["buy_elg_amount"] + q["buy_lg_amount"])
        / (q["buy_elg_vol"] + q["buy_lg_vol"])
        * 100
    )
    q["large_buy_pdev"] = _fillinf(abs(q["lbp"] - q["vwap"]) / q["vwap"], np.nan)
    weight = quote["amount"].fillna(0)
    return q["large_buy_pdev"].fillna(0).mul(weight).sum() / weight.sum()


def market_day_smallbuy_percentage(
    quote: pd.DataFrame, moneyflow: pd.DataFrame, **kwargs
) -> float:
    q = quote.join(moneyflow.loc[:, ["buy_sm_amount"]])
    q["small_buy_pct"] = _fillinf((q["buy_sm_amount"]) / q["amount"] * 10, 0)
    weight = quote["amount"].fillna(0)
    return q["small_buy_pct"].fillna(0).mul(weight).sum() / weight.sum()


def market_day_sqrt_avg_size(
    quote: pd.DataFrame, min: pd.DataFrame, moneyflow: pd.DataFrame, **kwargs
) -> float:
    if not min.empty and "num_trades" in min.columns:
        num_trades = min.groupby("secid")["num_trades"].sum()
    else:
        avg_size = [5, 20, 100, 500]
        mf_buy_size = moneyflow.loc[
            :, ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"]
        ]
        mf_buy_num = (mf_buy_size / avg_size).sum(axis=1)

        mf_sell_size = moneyflow.loc[
            :, ["sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount"]
        ]
        mf_sell_num = (mf_sell_size / avg_size).sum(axis=1)

        num_trades = (mf_buy_num + mf_sell_num) / 2 * 10
        num_trades = num_trades.rename("num_trades")

    q = quote.join(num_trades)
    return (_fillinf(q["amount"], 0).sum() / _fillinf(q["num_trades"], 0).sum()) ** 0.5


def market_day_open_close_percentage(
    quote: pd.DataFrame, min: pd.DataFrame, **kwargs
) -> float:
    ocamount = (
        min.query("minute <= 5 or minute >= 42")
        .groupby("secid")["amount"]
        .sum()
        .rename("open_close_amount")
    )
    q = quote.join(ocamount)
    q["open_close_pct"] = _fillinf(q["open_close_amount"] / q["amount"] / 1000, 0)
    weight = quote["amount"].fillna(0)
    return q["open_close_pct"].fillna(0).mul(weight).sum() / weight.sum()


def market_day_5min_ret_volatility(
    quote: pd.DataFrame, min: pd.DataFrame, val: pd.DataFrame, **kwargs
) -> float:
    returns = pandas_market_returns(quote, val, min)
    return returns.loc[returns.index >= 1].std(ddof=1)


def market_day_5min_ret_skewness(
    quote: pd.DataFrame, min: pd.DataFrame, val: pd.DataFrame, **kwargs
):
    returns = pandas_market_returns(quote, val, min)
    return returns.loc[returns.index >= 1].skew()


MINUTE_VERSION = "market-risk-grouped-shift-v2"
MINUTE_METRICS = (
    "market_day_sqrt_avg_size",
    "market_day_open_close_percentage",
    "market_day_5min_ret_volatility",
    "market_day_5min_ret_skewness",
)


def _minute_rows(frame):
    frame = frame.copy() if "secid" in frame else frame.reset_index()
    if "secid" not in frame or frame.secid.isna().any():
        raise ValueError("Missing security identifiers")
    return frame


def _minute_inputs(quote, val, minutes):
    q, v, m = map(_minute_rows, (quote, val, minutes))
    keys = ["secid"] + (["date"] if "date" in m else []) + ["minute"]
    if m.empty or q.empty or v.empty:
        raise ValueError("Empty required daily/minute source")
    if m[keys].isna().any().any() or m.duplicated(keys).any():
        raise ValueError("Missing/duplicate security/date/minute")
    if "date" in m and m.date.nunique() != 1:
        raise ValueError("Expected one trading date per market aggregation")
    if q.secid.duplicated().any() or v.secid.duplicated().any():
        raise ValueError("Duplicate daily security identifiers")
    return q, v, m


def pandas_market_returns(quote, val, minutes):
    q, v, m = _minute_inputs(quote, val, minutes)
    q, v = q.set_index("secid"), v.set_index("secid")
    keys = ["secid"] + (["date"] if "date" in m else [])
    m = m.sort_values(keys + ["minute"])
    # No cross-security/cross-date return and no implicit forward filling.
    m["ret"] = m.close / m.groupby(keys, sort=False).close.shift(1) - 1
    weight = (v.float_share * q.preclose).fillna(0).rename("weight")
    m = m.join(weight, on="secid")
    m["weighted"] = m.ret * m.weight
    aggregated = m.groupby("minute")[["weighted", "weight"]].sum()
    return (aggregated.weighted / aggregated.weight).rename("ret")


def polars_minutes(quote, val, minutes, moneyflow):
    """Recompute returns, trade counts, open/close fraction and moments in Polars."""
    import polars as pl

    q, v = [pl.from_pandas(_minute_rows(f), nan_to_null=True) for f in (quote, val)]
    m = minutes.fill_nan(None) if isinstance(minutes, pl.DataFrame) else pl.from_pandas(_minute_rows(minutes))
    key_columns = ["secid", "minute", *(["date"] if "date" in m else [])]
    if m.is_empty() or q.is_empty() or v.is_empty():
        raise ValueError("Empty required daily/minute source")
    identities = m.select(key_columns)
    if any(identities.null_count().row(0)) or identities.is_duplicated().any():
        raise ValueError("Missing/duplicate security/date/minute")
    if "date" in m and m["date"].n_unique() != 1:
        raise ValueError("Expected one trading date per market aggregation")
    if q["secid"].is_duplicated().any() or v["secid"].is_duplicated().any():
        raise ValueError("Duplicate daily security identifiers")
    keys = ["secid"] + (["date"] if "date" in m.columns else [])
    m = m.sort(keys + ["minute"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(keys) - 1)
        .fill_nan(None)
        .alias("ret")
    )
    weights = (
        q.select("secid", "preclose")
        .join(v.select("secid", "float_share"), on="secid", how="left")
        .with_columns(
            (pl.col("preclose") * pl.col("float_share"))
            .fill_nan(None)
            .fill_null(0)
            .alias("weight")
        )
    )
    aggregated = (
        m.join(weights.select("secid", "weight"), on="secid", how="left")
        .group_by("minute")
        .agg(
            (pl.col("ret") * pl.col("weight")).fill_nan(None).sum().alias("weighted"),
            pl.col("weight").sum(),
        )
    )
    aggregated = aggregated.with_columns(
        (pl.col("weighted") / pl.col("weight")).fill_nan(None).alias("ret")
    ).sort("minute")
    ret = aggregated.filter(pl.col("minute") >= 1)
    stats = ret.select(
        pl.col("ret").std(ddof=1).alias("std"),
        pl.col("ret").skew(bias=False).alias("skew"),
    ).row(0, named=True)
    if (
        ret.select(pl.col("ret").count()).item() >= 3
        and ret.select(pl.col("ret").max() - pl.col("ret").min()).item() == 0
    ):
        stats["skew"] = 0.0  # pandas' constant-sample convention
    oc = (
        m.filter((pl.col("minute") <= 5) | (pl.col("minute") >= 42))
        .group_by("secid")
        .agg(pl.col("amount").sum().alias("oc"))
    )
    joined = (
        q.select("secid", "amount")
        .join(oc, on="secid", how="left")
        .with_columns((pl.col("oc") / pl.col("amount") / 1000).alias("ratio"))
    )
    joined = joined.with_columns(
        pl.when(pl.col("ratio").is_finite())
        .then(pl.col("ratio"))
        .otherwise(0)
        .alias("ratio"),
        pl.col("amount").fill_null(0),
    )
    denominator = joined.select(pl.col("amount").sum()).item()
    oc_value = (
        joined.select((pl.col("ratio") * pl.col("amount")).fill_nan(None).sum()).item()
        / denominator
        if denominator
        else np.nan
    )
    if "num_trades" in m.columns:
        count = m.group_by("secid").agg(pl.col("num_trades").sum().alias("count"))
    else:
        mf = pl.from_pandas(_minute_rows(moneyflow), nan_to_null=True)
        if mf.select(pl.col("secid").is_duplicated().any()).item():
            raise ValueError("Duplicate moneyflow identifiers")
        count = mf.select(
            "secid",
            (
                pl.sum_horizontal(
                    [
                        (pl.col(f"{side}_{size}_amount") / scale).fill_null(0)
                        for side in ("buy", "sell")
                        for size, scale in zip(
                            ("sm", "md", "lg", "elg"), (5, 20, 100, 500)
                        )
                    ]
                )
                / 2
                * 10
            ).alias("count"),
        )
    counts = q.select("secid", "amount").join(count, on="secid", how="left")
    sums = counts.select(
        *[
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0).sum().alias(c)
            for c in ("amount", "count")
        ]
    ).row(0, named=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.sqrt(np.divide(sums["amount"], sums["count"]))
    metrics = dict(zip(MINUTE_METRICS, [size, oc_value, stats["std"], stats["skew"]]))
    metrics = {k: np.nan if v is None else float(v) for k, v in metrics.items()}
    return metrics, aggregated.select("minute", "ret").to_pandas().set_index(
        "minute"
    ).ret
