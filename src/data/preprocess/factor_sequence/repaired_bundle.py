"""Bounded-memory construction of versioned, masked repair data bundles."""

from __future__ import annotations

from dataclasses import asdict
from collections.abc import Callable, Mapping
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

from src.data.preprocess.factor_sequence.schema import BuildConfig, EVENTS
from src.data.preprocess.factor_sequence.numeric import stock_block, group_metrics, pack
from src.data.preprocess.factor_sequence.context_audit import candidate_market, MARKET_NAMES, RISK_NAMES
from src.data.preprocess.factor_sequence.bundle import file_digest, write_json
from src.proj.db.io.snapshot import snapshot_read_path

__all__ = ["REPAIR_VERSION", "build_repaired_data", "load_repaired_manifest"]
REPAIR_VERSION = "factor-sequence-repair-2"


def load_repaired_manifest(path, verify=True):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != REPAIR_VERSION:
        raise ValueError("Unsupported repair bundle")
    if verify:
        for name, digest in manifest["files"].items():
            if file_digest(path / name) != digest:
                raise ValueError(f"Corrupt repair bundle: {name}")
    return manifest


def build_repaired_data(root: str | Path, start: int, end: int, batch_size: int = 32, *, cfg: BuildConfig,
                        factor_sources: Mapping[str, tuple[str, str]],
                        read_factor: Callable[[str, int, np.ndarray, int], tuple[pd.Series, dict]],
                        read_event: Callable[[str], tuple[pd.DataFrame, Path]],
                        input_provenance: Callable[[], dict] | None = None) -> Path:
    """Read one date at a time; stock features are disk-backed, not a full RAM cube."""
    from src.proj import CALENDAR, DB
    from src.data import DATAVENDOR

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stage = root / "building"
    if stage.exists():
        shutil.rmtree(stage)  # private incomplete output, never a published version
    stage.mkdir()
    factors = cfg.factors
    # Original dataset begins in 2011; earlier events/risk are loaded independently.
    dates = np.asarray(CALENDAR.range(start, end, "td"), dtype=int)
    if not len(dates):
        raise ValueError("Empty data interval")
    desc = DATAVENDOR.INFO.get_list_dt(offset=91, keep_columns=["list_dt", "delist_dt"])
    if desc.empty or desc.index.has_duplicates:
        raise ValueError("Invalid independent stock universe")
    secids = desc.index.to_numpy()
    n, t, k = len(secids), len(dates), len(factors)
    names = [
        f"{channel}__{f}"
        for channel in (
            "value",
            "valid",
            "observed",
            "gap",
            "never",
            "change",
            "change_valid",
            "source_age",
            "source_age_valid",
            "degenerate",
        )
        for f in factors
    ]
    names += ["dtop__source_observed", "dtop__source_filled"]
    values = np.lib.format.open_memmap(
        stage / "factorseq.npy", mode="w+", dtype="float32", shape=(n, t, 1, len(names))
    )
    perf = np.full((t, k, 2), np.nan)
    state, previous = {}, None
    source_hash = hashlib.sha256()
    source_hash.update(desc.to_json().encode())
    sources_file = (stage / "sources.jsonl").open("w")
    groups_file = (stage / "groups.jsonl").open("w")
    try:
        for begin in range(0, t, batch_size):
            window = dates[begin : begin + batch_size]
            raw = np.full((n, len(window), k), np.nan)
            provenance = np.full((n, len(window), 2), np.nan)
            eligible = (window[None, :] >= desc.list_dt.to_numpy()[:, None]) & (
                window[None, :] < desc.delist_dt.to_numpy()[:, None]
            )
            for offset, date in enumerate(window):
                for j, factor in enumerate(factors):
                    name = factor.lstrip("+-")
                    series, record = read_factor(
                        name, int(date), secids, start
                    )
                    payload = json.dumps(record, sort_keys=True) + "\n"
                    sources_file.write(payload)
                    source_hash.update(payload.encode())
                    if record["status"] != "available":
                        raise ValueError(
                            f"Required factor source {name}@{date}: {record['status']}"
                        )
                    raw[:, offset, j] = series.to_numpy()
                    if name == "dtop":
                        source, key = factor_sources[name]
                        frame = DB.load(source, key, int(date))
                        required = ["dtop__source_observed", "dtop__source_filled"]
                        if not set(required) <= set(frame):
                            raise ValueError(
                                f"dtop@{date}: rebuild provenance before data blocks"
                            )
                        provenance[:, offset] = (
                            frame.set_index("secid")[required]
                            .reindex(secids)
                            .to_numpy()
                        )
                signed = raw[:, offset] * np.array(
                    [-1 if f.startswith("-") else 1 for f in factors]
                )
                signed[~eligible[:, offset]] = np.nan
                if previous is not None:
                    prices = DATAVENDOR.TRADE.get_specific_data(
                        int(CALENDAR.td(int(date), -1)),
                        int(date),
                        "adj",
                        "close",
                        prev=False,
                        pivot=True,
                    )
                    prices = prices.reindex(
                        index=[int(CALENDAR.td(int(date), -1)), int(date)],
                        columns=secids,
                    )
                    ret = prices.pct_change(fill_method=None).iloc[-1].to_numpy()
                    from src.shared.numeric.factor_groups import factor_groups

                    for j, factor in enumerate(factors):
                        metric = group_metrics(previous[:, j], ret, cfg.min_performance)
                        perf[begin + offset, j] = metric[[0, 3]] * [1, 100]
                        valid = np.isfinite(previous[:, j]) & np.isfinite(ret)
                        group = factor_groups(previous[valid, j])
                        _, counts = np.unique(group, return_counts=True)
                        reason = (
                            "insufficient_samples"
                            if valid.sum() < cfg.min_performance
                            else "fewer_than_two_groups"
                            if len(counts) < 2
                            else "valid"
                        )
                        groups_file.write(
                            json.dumps(
                                dict(
                                    date=int(date),
                                    factor=factor,
                                    counts=counts.tolist(),
                                    reason=reason,
                                )
                            )
                            + "\n"
                        )
                previous = signed
            block, _, _ = stock_block(raw, eligible, cfg, state=state)
            values[:, begin : begin + len(window), :, : k * 10] = block
            values[:, begin : begin + len(window), 0, k * 10 :] = np.nan_to_num(
                provenance
            )
            values.flush()
            DATAVENDOR.TRADE.clear_all()
        # Keep all prehistory for event ages and 120-day risk maturity.
        histories = {}
        for event in EVENTS:
            frame, path = read_event(event)
            if frame.empty or frame.date.duplicated().any():
                raise ValueError(f"Invalid event history: {event}")
            histories[event] = frame.set_index("date")[event]
            source_hash.update(file_digest(path).encode())
        risk = DB.load("market_daily", "risk")
        quote = DB.load("index_daily_ts", "000985.CSI")
        for frame in (risk, quote):
            if frame.empty:
                raise ValueError("Missing risk/index source")
        for source, key in [("market_daily", "risk"), ("index_daily_ts", "000985.CSI")]:
            source_hash.update(file_digest(snapshot_read_path(DB.path(source, key))).encode())
        early = min(
            int(dates[0]),
            *(int(s.index.min()) for s in histories.values()),
            int(risk.date.min()),
        )
        market_dates = np.asarray(CALENDAR.range(early, end, "td"), dtype=int)
        market = pd.DataFrame(index=market_dates)
        for event, series in histories.items():
            market[event] = series.reindex(market_dates)
        market["market_return"] = (
            quote.set_index("trade_date").pct_chg.reindex(market_dates) / 100
        )
        for name in RISK_NAMES:
            market[name] = risk.set_index("date")[name].reindex(market_dates)
        slim_market = candidate_market(market).reindex(dates)
        m, mnames = pack({name: slim_market[name].to_numpy() for name in MARKET_NAMES})
        p, pnames = pack(
            {
                f"{factor.lstrip('+-')}__{metric}": perf[:, j, i]
                for j, factor in enumerate(factors)
                for i, metric in enumerate(("ic", "spread"))
            }
        )
        np.save(stage / "factorperf.npy", p[None, :, None, :])
        np.save(stage / "factormarket.npy", m[None, :, None, :])
        np.save(stage / "dates.npy", dates)
        np.save(stage / "secids.npy", secids)
        del values
    finally:
        sources_file.close()
        groups_file.close()
    from src.data.preprocess.factor_sequence.context_audit import dictionary

    feature_dictionary = dictionary()
    feature_dictionary["input_features"] = {
        "factorseq": names,
        "factorperf": pnames,
        "factormarket": mnames,
    }
    write_json(stage / "feature_dictionary.json", feature_dictionary)
    files = {p.name: file_digest(p) for p in stage.iterdir() if p.is_file()}
    from src.shared.numeric.factor_groups import GROUPING_VERSION

    from src.data.preprocess.factor_sequence.provenance import component_paths
    code = {name: file_digest(p) for name, p in component_paths().items()}
    identity = dict(
        recovery=input_provenance() if input_provenance is not None else None,
        grouping=GROUPING_VERSION,
        transform_hashes=code,
        format=REPAIR_VERSION,
        config=asdict(cfg),
        sources=source_hash.hexdigest(),
        files=files,
    )
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    blocks = {
        key: dict(
            features=features,
            shared=key != "factorseq",
            shape=[n if key == "factorseq" else 1, t, 1, len(features)],
        )
        for key, features in [
            ("factorseq", names),
            ("factorperf", pnames),
            ("factormarket", mnames),
        ]
    }
    manifest = dict(
        identity,
        version=version,
        schema_version=2,
        config_hash=cfg.fingerprint,
        dates=[int(dates[0]), int(dates[-1])],
        blocks=blocks,
        limitations=[
            "Historical source revisions not reconstructed",
            "Industry membership history unavailable",
        ],
    )
    write_json(stage / "manifest.json", manifest)
    target = root / version
    if target.exists():
        load_repaired_manifest(target)
        shutil.rmtree(stage)
    else:
        stage.rename(target)
    return target
