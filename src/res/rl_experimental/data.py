from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np


PANEL_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PanelData:
    """Point-in-time panel used by the environment.

    The first seven arrays are the version-1, single-period contract. A real
    snapshot also supplies execution arrays: decide at ``decision_dates[t]``
    close, execute at ``execution_dates[t]`` open, and value at the return-end
    close.
    """

    stock_ids: np.ndarray
    decision_dates: np.ndarray
    return_end_dates: np.ndarray
    features: np.ndarray
    alpha: np.ndarray
    investable: np.ndarray
    forward_returns: np.ndarray
    execution_dates: np.ndarray | None = None
    overnight_returns: np.ndarray | None = None
    intraday_returns: np.ndarray | None = None
    can_buy: np.ndarray | None = None
    can_sell: np.ndarray | None = None
    valuation_stale: np.ndarray | None = None
    industry_codes: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    schema_version: int = PANEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        stock_ids = np.asarray(self.stock_ids, dtype=np.int64)
        dates = np.asarray(self.decision_dates, dtype=np.int64)
        end_dates = np.asarray(self.return_end_dates, dtype=np.int64)
        features = np.asarray(self.features, dtype=np.float32)
        alpha = np.asarray(self.alpha, dtype=np.float32)
        investable = np.asarray(self.investable, dtype=bool)
        returns = np.asarray(self.forward_returns, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError("features must have shape [time, stock, feature]")
        t, n, f = features.shape
        if stock_ids.shape != (n,) or len(np.unique(stock_ids)) != n:
            raise ValueError("stock_ids must be a unique [stock] array")
        for name, value in {"alpha": alpha, "investable": investable, "forward_returns": returns}.items():
            if value.shape != (t, n):
                raise ValueError(f"{name} must have shape [time, stock]")
        if dates.shape != (t,) or end_dates.shape != (t,):
            raise ValueError("date arrays must have shape [time]")
        if np.any(np.diff(dates) <= 0) or np.any(end_dates < dates):
            raise ValueError("decision dates must increase and return dates cannot precede them")
        if not np.isfinite(features).all() or not np.isfinite(alpha).all():
            raise ValueError("features and alpha must be finite")
        if self.feature_names and len(self.feature_names) != f:
            raise ValueError("feature_names must match the feature dimension")

        execution_values = (
            self.execution_dates,
            self.overnight_returns,
            self.intraday_returns,
            self.can_buy,
            self.can_sell,
            self.valuation_stale,
        )
        if any(value is not None for value in execution_values) and not all(value is not None for value in execution_values):
            raise ValueError("real execution fields must either all be present or all be absent")
        if self.is_execution_panel:
            execution_dates = np.asarray(self.execution_dates, dtype=np.int64)
            overnight = np.asarray(self.overnight_returns, dtype=np.float32)
            intraday = np.asarray(self.intraday_returns, dtype=np.float32)
            can_buy = np.asarray(self.can_buy, dtype=bool)
            can_sell = np.asarray(self.can_sell, dtype=bool)
            stale = np.asarray(self.valuation_stale, dtype=bool)
            if execution_dates.shape != (t,) or np.any(execution_dates <= dates) or np.any(end_dates < execution_dates):
                raise ValueError("execution dates must be after decisions and no later than return ends")
            for name, value in {
                "overnight_returns": overnight,
                "intraday_returns": intraday,
                "can_buy": can_buy,
                "can_sell": can_sell,
                "valuation_stale": stale,
            }.items():
                if value.shape != (t, n):
                    raise ValueError(f"{name} must have shape [time, stock]")
            if np.any(~np.isfinite(overnight[investable & can_buy])) or np.any(~np.isfinite(intraday[investable & can_buy])):
                raise ValueError("buyable stocks require finite execution returns")
            object.__setattr__(self, "execution_dates", execution_dates)
            object.__setattr__(self, "overnight_returns", overnight)
            object.__setattr__(self, "intraday_returns", intraday)
            object.__setattr__(self, "can_buy", can_buy)
            object.__setattr__(self, "can_sell", can_sell)
            object.__setattr__(self, "valuation_stale", stale)
        elif np.any(~np.isfinite(returns[investable])):
            raise ValueError("forward returns must be finite for investable stocks")

        if self.industry_codes is not None:
            industry = np.asarray(self.industry_codes, dtype=np.int16)
            if industry.shape != (t, n) or np.any(industry < -1):
                raise ValueError("industry_codes must be [time, stock] with -1 for missing")
            object.__setattr__(self, "industry_codes", industry)
        object.__setattr__(self, "stock_ids", stock_ids)
        object.__setattr__(self, "decision_dates", dates)
        object.__setattr__(self, "return_end_dates", end_dates)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "investable", investable)
        object.__setattr__(self, "forward_returns", returns)
        object.__setattr__(self, "feature_names", tuple(str(x) for x in self.feature_names))

    @property
    def is_execution_panel(self) -> bool:
        return self.execution_dates is not None

    def execution_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the open-execution arrays stored together by ``__post_init__``."""
        execution_dates = self.execution_dates
        overnight_returns = self.overnight_returns
        intraday_returns = self.intraday_returns
        can_buy = self.can_buy
        can_sell = self.can_sell
        valuation_stale = self.valuation_stale
        if (
            execution_dates is None
            or overnight_returns is None
            or intraday_returns is None
            or can_buy is None
            or can_sell is None
            or valuation_stale is None
        ):
            raise RuntimeError("execution arrays are absent")
        return execution_dates, overnight_returns, intraday_returns, can_buy, can_sell, valuation_stale

    @property
    def n_steps(self) -> int:
        return self.features.shape[0]

    @property
    def n_stocks(self) -> int:
        return self.features.shape[1]

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    @property
    def n_industries(self) -> int:
        if self.industry_codes is None or not np.any(self.industry_codes >= 0):
            return 0
        return int(self.industry_codes.max()) + 1

    @property
    def observation_feature_count(self) -> int:
        return self.n_features + self.n_industries

    def observation_features(self, time_index: int, stock_indices: np.ndarray) -> np.ndarray:
        values = self.features[time_index, stock_indices]
        industry = self.industry_codes
        if industry is None or self.n_industries == 0:
            return values
        one_hot = np.zeros((len(stock_indices), self.n_industries), dtype=np.float32)
        codes = industry[time_index, stock_indices]
        valid = codes >= 0
        one_hot[np.flatnonzero(valid), codes[valid]] = 1.0
        return np.concatenate([values, one_hot], axis=1)

    def contract(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        values = [
            self.stock_ids,
            self.decision_dates,
            self.return_end_dates,
            self.features,
            self.alpha,
            self.investable,
            self.forward_returns,
        ]
        values.extend(
            value
            for value in (
                self.execution_dates,
                self.overnight_returns,
                self.intraday_returns,
                self.can_buy,
                self.can_sell,
                self.valuation_stale,
                self.industry_codes,
            )
            if value is not None
        )
        for value in values:
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        digest.update(json.dumps(list(self.feature_names), ensure_ascii=False).encode())
        return {
            "schema_version": int(self.schema_version),
            "execution_mode": "open" if self.is_execution_panel else "single_period",
            "n_features": self.n_features,
            "n_industries": self.n_industries,
            "feature_names": list(self.feature_names),
            "identity_digest": digest.hexdigest(),
        }

    def save_npz(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "stock_ids": self.stock_ids,
            "decision_dates": self.decision_dates,
            "return_end_dates": self.return_end_dates,
            "features": self.features,
            "alpha": self.alpha,
            "investable": self.investable,
            "forward_returns": self.forward_returns,
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "schema_version": np.asarray(self.schema_version, dtype=np.int64),
        }
        for name in (
            "execution_dates",
            "overnight_returns",
            "intraday_returns",
            "can_buy",
            "can_sell",
            "valuation_stale",
            "industry_codes",
        ):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        # A plain dict[str, ndarray] can contain the keyword-only name allow_pickle.
        save = cast(Callable[..., None], np.savez_compressed)
        save(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> "PanelData":
        with np.load(path, allow_pickle=False) as data:
            kwargs = {name: data[name] for name in data.files}
        if "feature_names" in kwargs:
            kwargs["feature_names"] = tuple(kwargs["feature_names"].tolist())
        if "schema_version" in kwargs:
            kwargs["schema_version"] = int(kwargs["schema_version"])
        return cls(**kwargs)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, panel: PanelData, start: int, end: int) -> "Standardizer":
        if not 0 <= start < end <= panel.n_steps:
            raise ValueError("invalid fit interval")
        values = panel.features[start:end][panel.investable[start:end]]
        if values.size == 0:
            raise ValueError("training interval contains no investable observations")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        if panel.feature_names:
            indicators = np.asarray([name.endswith("_missing") for name in panel.feature_names])
            mean[indicators] = 0.0
            scale[indicators] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, panel: PanelData) -> PanelData:
        if self.mean.shape != (panel.n_features,) or self.scale.shape != (panel.n_features,):
            raise ValueError("normalizer does not match panel feature schema")
        features = (panel.features - self.mean) / self.scale
        return replace(panel, features=features.astype(np.float32))

    def save_npz(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale)


def chronological_split(n_steps: int, train: float = 0.6, valid: float = 0.2) -> dict[str, tuple[int, int]]:
    if n_steps < 5 or not 0 < train < 1 or not 0 < valid < 1 or train + valid >= 1:
        raise ValueError("need at least five rows and valid split fractions")
    train_end = max(1, int(n_steps * train))
    valid_end = max(train_end + 1, int(n_steps * (train + valid)))
    valid_end = min(valid_end, n_steps - 1)
    return {"train": (0, train_end), "valid": (train_end, valid_end), "test": (valid_end, n_steps)}


def date_split(panel: PanelData, train_end_date: int | None = None, valid_end_date: int | None = None) -> dict[str, tuple[int, int]]:
    """Create contiguous splits while keeping every return inside its boundary."""
    if train_end_date is None and valid_end_date is None:
        rough = chronological_split(panel.n_steps)
        train_end_date = int(panel.decision_dates[rough["train"][1]])
        valid_end_date = int(panel.decision_dates[rough["valid"][1]])
    if train_end_date is None or valid_end_date is None or train_end_date >= valid_end_date:
        raise ValueError("train_end_date and valid_end_date must both be set and increasing")
    train_end = int(np.searchsorted(panel.return_end_dates, train_end_date, side="right"))
    valid_start = int(np.searchsorted(panel.decision_dates, train_end_date, side="left"))
    valid_end = int(np.searchsorted(panel.return_end_dates, valid_end_date, side="right"))
    test_start = int(np.searchsorted(panel.decision_dates, valid_end_date, side="left"))
    splits = {"train": (0, train_end), "valid": (valid_start, valid_end), "test": (test_start, panel.n_steps)}
    if any(end - start < 1 for start, end in splits.values()):
        raise ValueError(f"date boundaries produce an empty split: {splits}")
    return splits


def make_synthetic_panel(n_steps: int = 320, n_stocks: int = 200, n_features: int = 8, seed: int = 7) -> PanelData:
    """Create a small, weakly predictable panel for end-to-end smoke tests."""
    if min(n_steps, n_stocks, n_features) <= 1:
        raise ValueError("synthetic dimensions must all exceed one")
    rng = np.random.default_rng(seed)
    dates = np.arange(n_steps, dtype=np.int64)
    stock_ids = np.arange(10_000, 10_000 + n_stocks, dtype=np.int64)
    industries = rng.integers(0, max(2, min(10, n_stocks // 5)), size=n_stocks)
    market = rng.normal(0.0, 0.008, size=n_steps)
    industry = rng.normal(0.0, 0.006, size=(n_steps, industries.max() + 1))
    latent = rng.normal(size=(n_steps, n_stocks, n_features)).astype(np.float32)
    latent[1:] = 0.75 * latent[:-1] + 0.25 * latent[1:]
    alpha = (0.7 * latent[:, :, 0] - 0.3 * latent[:, :, 1] + rng.normal(0, 0.7, (n_steps, n_stocks))).astype(np.float32)
    returns = (market[:, None] + industry[:, industries] + 0.0015 * alpha + rng.normal(0.0, 0.018, (n_steps, n_stocks))).astype(np.float32)
    listing_start = rng.integers(0, max(1, n_steps // 8), size=n_stocks)
    delisting = rng.integers(max(2, n_steps * 7 // 8), n_steps + 1, size=n_stocks)
    investable = (dates[:, None] >= listing_start) & (dates[:, None] < delisting)
    investable &= rng.random((n_steps, n_stocks)) > 0.01
    returns[~investable] = np.nan
    return PanelData(
        stock_ids=stock_ids,
        decision_dates=dates,
        return_end_dates=dates + 1,
        features=latent,
        alpha=alpha,
        investable=investable,
        forward_returns=returns,
        feature_names=tuple(f"synthetic_{i}" for i in range(n_features)),
    )
