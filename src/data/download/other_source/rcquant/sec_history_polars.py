"""Vectorized validation and reconstruction of historical RCQuant equity bars."""
from __future__ import annotations

from dataclasses import dataclass
from math import ulp

import polars as pl

from .migration_factors import LEGACY_PRICE_HALF_UNIT, PRICE, VWAP_EPSILON

ID = 'order_book_id'
STOCK_TIME = [ID, 'datetime']
TOTALS = ['volume', 'amount', 'num_trades']
QUOTE = [*PRICE, *TOTALS]
REFERENCE = [f'{name}_day' for name in [*PRICE, 'volume']]
RATIO = 'ratio'
CHECK_EPSILON = 1e-8
LEGACY_VOLUME_HALF_UNIT = .00005
DAILY_VOLUME_TOLERANCE = .100001


def _event_date(column: str) -> pl.Expr:
    return pl.col(column).dt.strftime('%Y%m%d').cast(pl.Int64).alias('effective')


@dataclass
class FactorHistory:
    price: pl.DataFrame
    split: pl.DataFrame

    @classmethod
    def from_events(cls, price: pl.DataFrame, split: pl.DataFrame) -> FactorHistory:
        if price.is_empty():
            price = pl.DataFrame(schema={ID: pl.String, 'effective': pl.Int64,
                                         'factor': pl.Float64, 'before': pl.Float64})
        else:
            price = (
                price.select(ID, _event_date('ex_date'), pl.col('ex_cum_factor').alias('factor'), 'ex_factor')
                .unique(maintain_order=True).sort(ID, 'effective')
            )
            if price.select(ID, 'effective').n_unique() != len(price):
                raise ValueError('Conflicting RCQuant price factors on the same ex-date')
            price = price.with_columns(
                (pl.col('factor') / pl.col('ex_factor')).first().over(ID).alias('before')
            ).drop('ex_factor')
        if split.is_empty():
            split = pl.DataFrame(schema=price.schema)
        else:
            split = (
                split.select(ID, _event_date('ex_dividend_date'), 'book_closure_date',
                             'split_coefficient_from', 'split_coefficient_to')
                .unique(maintain_order=True).sort(ID, 'effective')
                .select(ID, 'effective', (pl.col('split_coefficient_to') / pl.col('split_coefficient_from'))
                        .cum_prod().over(ID).alias('factor'), pl.lit(1.).alias('before'))
            )
        return cls(price, split)

    @staticmethod
    def _at(events: pl.DataFrame, date: int) -> pl.DataFrame:
        return events.group_by(ID).agg(
            pl.col('factor').filter(pl.col('effective') <= date).last()
            .fill_null(pl.col('before').first()).alias('current')
        )

    def candidates(self, date: int, ids: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        base = ids.select(ID, pl.lit(1.).alias(RATIO))
        ratios = []
        for events, inverse in ((self.price, False), (self.split, True)):
            future = events.filter(pl.col('effective') > date).join(base.select(ID), on=ID)
            reference = self._at(events, date)
            ratio = pl.col('current') / pl.col('factor') if inverse else pl.col('factor') / pl.col('current')
            ratios.append(pl.concat([base, future.join(reference, on=ID).select(ID, ratio.alias(RATIO))]).unique())
        return ratios[0], ratios[1]

    def daily(self, date: int, stocks: pl.DataFrame) -> pl.DataFrame:
        return (
            stocks.join(self._at(self.price, date).rename({'current': 'ex_cum_factor'}), on=ID, how='left')
            .join(self._at(self.split, date).rename({'current': 'split_cum_factor'}), on=ID, how='left')
            .with_columns(pl.col('ex_cum_factor', 'split_cum_factor').fill_null(1.0), pl.lit(date).alias('date'))
            .select('date', ID, 'secid', 'ex_cum_factor', 'split_cum_factor')
        )


def daily_reference(stocks: pl.DataFrame, day: pl.DataFrame) -> pl.DataFrame:
    if day.is_empty():
        return pl.DataFrame(schema={ID: pl.String, **dict.fromkeys(REFERENCE, pl.Float64)})
    if day['secid'].n_unique() != len(day):
        raise ValueError('Duplicate secid in the Tushare daily reference')
    return stocks.join(day, on='secid').select(
        ID, *[pl.col(name).cast(pl.Float64).alias(f'{name}_day') for name in PRICE],
        (pl.col('volume').cast(pl.Float64) * 1000).alias('volume_day'),
    )


def daily_volume_mismatches(bars: pl.DataFrame, refs: pl.DataFrame) -> list[str]:
    reported = bars.group_by(ID).agg(pl.col('volume').sum()).join(refs.select(ID, 'volume_day'), on=ID)
    return reported.filter(
        pl.col('volume_day').is_finite()
        & (pl.col('volume') - pl.col('volume_day')).abs().gt(DAILY_VOLUME_TOLERANCE)
    )[ID].sort().to_list()


def _daily_quotes(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.group_by(ID).agg(
        pl.col('open').first(), pl.col('high').max(), pl.col('low').min(), pl.col('close').last(),
        pl.col('volume').sum(),
        pl.all_horizontal(pl.col([*PRICE, 'volume']).is_finite().fill_null(False)).all().alias('finite'),
        (pl.all_horizontal(pl.col(PRICE).gt(0)) & pl.col('volume').ge(0)).all().alias('valid'),
    )


def _round(expr: pl.Expr, digits: int) -> pl.Expr:
    scale = 10 ** digits
    scaled = expr * scale + .5
    # All quote inputs here are nonnegative. Allow one floating-point unit
    # at a decimal tie, consistent with the legacy nextafter calculation.
    return (scaled + scaled * (ulp(1.) / 2)).floor() / scale


def _unique_choice(raw: pl.DataFrame, options: pl.DataFrame, fields: list[str], digits: int,
                   output: str) -> pl.DataFrame:
    # With positive prices/quantities, equal results at both extremes imply
    # that any other admissible multiplier has the same rounded result.
    ranges = options.group_by(ID).agg(pl.col(RATIO).min().alias('small'), pl.col(RATIO).max().alias('large'))
    equivalent = (
        ranges.filter(pl.col('small') != pl.col('large')).join(raw.select(ID, *fields), on=ID)
        .group_by(ID).agg(pl.all_horizontal([
            _round(pl.col(name) * pl.col('small'), digits).eq(
                _round(pl.col(name) * pl.col('large'), digits)) for name in fields
        ]).all().alias('equivalent'))
    )
    return (
        ranges.join(equivalent, on=ID, how='left')
        .filter(pl.col('equivalent').fill_null(True)).select(ID, pl.col('small').alias(output))
    )


def restore_factors(raw: pl.DataFrame, refs: pl.DataFrame, history: FactorHistory,
                    date: int, stocks: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Calibrate on Tushare without comparing or fitting to provider minute bars."""
    price, volume = history.candidates(date, stocks.select(ID).unique())
    day = _daily_quotes(raw).join(refs, on=ID).filter(
        pl.col('finite') & pl.col('valid')
        & pl.all_horizontal(pl.col(REFERENCE).is_finite()) & pl.col('volume_day').gt(0)
    )
    price_candidates = day.join(price, on=ID).filter(pl.all_horizontal([
        (pl.col(name) * pl.col(RATIO) - pl.col(f'{name}_day')).abs().le(.00501) for name in PRICE
    ])).select(ID, RATIO, *REFERENCE[:4])

    prices = raw.select(ID, *PRICE).join(price_candidates, on=ID, maintain_order='left')
    price_valid = []
    price_rounded = []
    for name in PRICE:
        scaled = pl.col(name) * pl.col(RATIO)
        rounded = _round(scaled, 2)
        error = (rounded - scaled).abs()
        limit = LEGACY_PRICE_HALF_UNIT * pl.col(RATIO) + 1e-10
        price_valid.append((error <= limit) & ((.01 - error) > limit))
        price_rounded.append(rounded.alias(name))
    prices = (
        prices.with_columns(pl.all_horizontal(price_valid).alias('valid'), *price_rounded)
        .group_by(ID, RATIO).agg(pl.col('valid').all(), pl.col('open').first(), pl.col('high').max(),
                                pl.col('low').min(), pl.col('close').last(), pl.col(REFERENCE[:4]).first())
        .filter(pl.col('valid') & pl.all_horizontal([
            (pl.col(name) - pl.col(f'{name}_day')).abs().le(1e-5) for name in PRICE
        ])).select(ID, RATIO)
    )

    eligible_volume = day.join(volume, on=ID).filter(
        (pl.col('volume') * pl.col(RATIO) - pl.col('volume_day')).abs()
        <= DAILY_VOLUME_TOLERANCE + (LEGACY_VOLUME_HALF_UNIT + CHECK_EPSILON) * pl.col(RATIO) * 240 + CHECK_EPSILON
    ).select(ID, RATIO, 'volume_day')
    integer = _round(pl.col('volume') * pl.col(RATIO), 0)
    # A legacy pre-adjusted bar stores at most four decimals. Verify precision
    # in those units after rounding to an integer number of unadjusted shares.
    volumes = (
        raw.select(ID, 'volume').join(eligible_volume, on=ID)
        .with_columns(integer.alias('shares'),
                      (integer / pl.col(RATIO) - pl.col('volume')).abs()
                      .le(LEGACY_VOLUME_HALF_UNIT + CHECK_EPSILON).alias('valid'))
        .group_by(ID, RATIO).agg(pl.col('valid').all(), pl.col('shares').sum(), pl.col('volume_day').first())
        .filter(pl.col('valid') & (pl.col('shares') - pl.col('volume_day')).abs().le(DAILY_VOLUME_TOLERANCE))
        .select(ID, RATIO)
    )
    choices = _unique_choice(raw, prices, PRICE, 2, 'price_factor').join(
        _unique_choice(raw, volumes, ['volume'], 0, 'volume_factor'), on=ID,
    )
    output = raw.join(choices, on=ID).select(
        *STOCK_TIME,
        *[_round(pl.col(name) * pl.col('price_factor'), 2).alias(name) for name in PRICE],
        _round(pl.col('volume') * pl.col('volume_factor'), 0).alias('volume'), 'amount', 'num_trades',
    )
    return output, choices.filter((pl.col('volume_factor') - 1).abs() > 1e-12).height


def normalize_bars(raw: pl.DataFrame, stocks: pl.DataFrame) -> pl.DataFrame:
    clock = pl.col('datetime').dt.hour().cast(pl.Int64) * 60 + pl.col('datetime').dt.minute()
    minute = clock - 571 - pl.when(clock >= 781).then(90).otherwise(0)
    bars = raw.join(stocks.select('code', 'secid'), on='code', how='left').select(
        'secid', minute.alias('minute'), *QUOTE,
    )
    return with_vwap(bars.sort('secid', 'minute'), 'open').select(
        'secid', 'minute', *PRICE, 'amount', 'volume', 'vwap', 'num_trades',
    )


def with_vwap(bars: pl.DataFrame, fallback: str) -> pl.DataFrame:
    ratio = pl.col('amount') / pl.col('volume')
    return bars.with_columns(
        pl.when(pl.col('volume').eq(0) | ratio.is_infinite())
        .then(pl.when(pl.col(fallback).is_finite()).then(pl.col(fallback)))
        .when(pl.col('volume').gt(0) & ratio.is_finite()).then(ratio)
        .alias('vwap')
    )


def aggregate_bars(minute: pl.DataFrame, period: int) -> pl.DataFrame:
    fill = pl.coalesce(pl.col('close').fill_nan(None), pl.col('open').fill_nan(None))
    bars = (
        minute.with_columns(pl.col('minute') // period,
                            pl.col(PRICE).fill_nan(None).fill_null(fill))
        .group_by('secid', 'minute', maintain_order=True)
        .agg(pl.col('open').drop_nulls().first(), pl.col('high').max(), pl.col('low').min(),
             pl.col('close').drop_nulls().last(),
             pl.col(TOTALS).sum())
    )
    return with_vwap(bars, 'close').select(
        'secid', 'minute', *PRICE, 'amount', 'volume', 'num_trades', 'vwap',
    )


def vwap_audit(bars: pl.DataFrame, period: int) -> pl.DataFrame:
    vwap = pl.col('vwap')
    low, high = pl.col('low'), pl.col('high')
    outside = pl.col('volume').gt(0) & (
        (vwap < low - VWAP_EPSILON) | (vwap > high + VWAP_EPSILON) | ~vwap.is_finite().fill_null(False)
    )
    return (
        bars.filter(outside).select('secid', 'minute', *QUOTE, 'vwap')
        .with_columns(pl.lit(period).alias('period'), (vwap - vwap.clip(low, high)).alias('gap_yuan'))
    )
