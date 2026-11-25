"""Corporate actions and conservative inverse adjustment for legacy equity bars."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    'PRICE', 'Events', 'EMPTY_EVENTS', 'event_series', 'normalize_events',
    'inverted_prices', 'restore_stock_day', 'round_half_up', 'vwap_outside',
]

PRICE = ['open', 'high', 'low', 'close']
# Legacy get_price(pre) quotes observed at four decimal places. A half-unit
# interval includes the precision lost by the vendor before the price was stored.
LEGACY_PRICE_HALF_UNIT = .00005
VWAP_EPSILON = 1e-8


@dataclass(frozen=True)
class Events:
    dates: np.ndarray
    factors: np.ndarray
    before: float = 1.

    def on(self, date: int) -> float:
        index = self.dates.searchsorted(date, side='right') - 1
        return float(self.factors[index]) if index >= 0 else self.before

    def candidates(self, date: int) -> np.ndarray:
        """Distinct factors for a forward-adjustment base on or after date."""
        return np.r_[self.on(date), self.factors[self.dates > date]]


EMPTY_EVENTS = Events(np.array([], dtype=int), np.array([], dtype=float))
EVENT_COLUMNS = {
    'get_ex_factor': ['order_book_id', 'ex_date', 'ex_factor', 'ex_cum_factor'],
    'get_split': ['order_book_id', 'ex_dividend_date', 'book_closure_date',
                  'split_coefficient_from', 'split_coefficient_to'],
}


def normalize_events(method: str, ids: list[str], response) -> pd.DataFrame:
    if response is None or isinstance(response, pd.DataFrame) and response.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS[method])
    if not isinstance(response, pd.DataFrame):
        raise ValueError(f'{method}: invalid response type {type(response).__name__}')
    frame = response.reset_index()
    fields = EVENT_COLUMNS[method]
    if not set(fields).issubset(frame) or not frame['order_book_id'].isin(ids).all():
        raise ValueError(f'{method}: unexpected response fields or instrument IDs')
    event_date = fields[1]
    frame[event_date] = pd.to_datetime(frame[event_date], errors='raise')
    if frame[event_date].isna().any():
        raise ValueError(f'{method}: missing event date')
    values = fields[-2:]
    frame[values] = frame[values].apply(pd.to_numeric, errors='raise')
    if not (np.isfinite(frame[values]) & frame[values].gt(0)).all().all():
        raise ValueError(f'{method}: invalid adjustment factor')
    return frame


def event_series(method: str, frame: pd.DataFrame) -> dict[str, Events]:
    result = {}
    date_column = EVENT_COLUMNS[method][1]
    for code, rows in frame.groupby('order_book_id', sort=False):
        rows = rows.sort_values(date_column, kind='stable')
        if method == 'get_ex_factor':
            fields = ['ex_cum_factor', 'ex_factor']
            rows = rows.drop_duplicates([date_column, *fields])
            if rows[date_column].duplicated().any():
                raise ValueError(f'{code}: conflicting price factors on one ex-date')
            factors = rows['ex_cum_factor'].to_numpy(float)
            before = factors[0] / rows['ex_factor'].iloc[0]
        else:
            fields = ['split_coefficient_to', 'split_coefficient_from']
            rows = rows.drop_duplicates([date_column, 'book_closure_date', *fields])
            factors = (rows[fields[0]] / rows[fields[1]]).cumprod().to_numpy(float)
            before = 1.
        result[code] = Events(
            rows[date_column].dt.strftime('%Y%m%d').to_numpy(int), factors, before,
        )
    return result


def vwap_outside(frame: pd.DataFrame) -> np.ndarray:
    traded = frame['volume'].to_numpy(float) > 0
    vwap = frame['amount'].div(frame['volume'].where(traded)).to_numpy(float)
    return traded & (
        (vwap < frame['low'].to_numpy(float) - VWAP_EPSILON)
        | (vwap > frame['high'].to_numpy(float) + VWAP_EPSILON)
        | ~np.isfinite(vwap)
    )


def round_half_up(values: np.ndarray, scale: int) -> np.ndarray:
    # Prices and shares are nonnegative. nextafter compensates for a binary
    # approximation immediately below an exact decimal half-cent.
    return np.floor(np.nextafter(values * scale + .5, np.inf)) / scale


def inverted_prices(prices: np.ndarray, multiplier: float) -> np.ndarray | None:
    restored = prices * multiplier
    rounded = round_half_up(restored, 100)
    error = LEGACY_PRICE_HALF_UNIT * multiplier + 1e-10
    if np.any(np.abs(rounded - restored) > error):
        return None
    # Reject any inverse interval that also permits another cent price.
    if np.any(.01 - np.abs(rounded - restored) <= error):
        return None
    return rounded


def restore_stock_day(
    prices: np.ndarray, volume: np.ndarray, daily: np.ndarray, date: int,
    factors: Events, splits: Events,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Use native daily references to select a unique candidate for price and volume.

    Unknown vendor bar boundaries and vendor corrections can cause the daily
    check to fail. Callers then obtain the stock-day directly in none mode.
    """
    if not np.isfinite(prices).all() or not np.isfinite(volume).all() or not np.isfinite(daily).all():
        return None
    if np.any(prices <= 0) or np.any(volume < 0) or daily[-1] <= 0:
        return None
    endpoints = np.array([prices[0, 0], prices[:, 1].max(), prices[:, 2].min(), prices[-1, 3]])
    price_today = factors.on(date)
    price_options = {}
    for multiplier in np.unique(factors.candidates(date) / price_today):
        if not np.allclose(endpoints * multiplier, daily[:4], atol=.00501, rtol=0):
            continue
        restored = inverted_prices(prices, multiplier)
        if restored is not None and np.allclose(
            np.array([restored[0, 0], restored[:, 1].max(), restored[:, 2].min(), restored[-1, 3]]),
            daily[:4], rtol=0, atol=1e-5,
        ):
            price_options[restored.tobytes()] = restored
    if len(price_options) != 1:
        return None

    volume_options = {}
    for multiplier in np.unique(splits.on(date) / splits.candidates(date)):
        if abs(volume.sum() * multiplier - daily[-1]) > 1:
            continue
        shares = round_half_up(volume * multiplier, 1)
        if np.max(np.abs(shares - volume * multiplier)) > 1e-5:
            continue
        if abs(shares.sum() - daily[-1]) <= .100001:
            volume_options.setdefault(shares.tobytes(), (shares, multiplier != 1.))
    if len(volume_options) != 1:
        return None
    shares, scaled = next(iter(volume_options.values()))
    return next(iter(price_options.values())), shares, scaled
