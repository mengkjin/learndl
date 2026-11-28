"""Freeze a repaired dataset and label plan for comparable ordinary schedules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from .label_snapshot import graph_label_snapshot, label_digest, read_live_labels

import numpy as np

from src.data.preprocess.factor_sequence.schema import FACTORS, STOCK_CHANNELS
from src.data.preprocess.factor_sequence.context_audit import MARKET_NAMES, PERF_NAMES
from src.data.preprocess.factor_sequence.repaired_bundle import load_repaired_manifest
from src.data.preprocess.factor_sequence.bundle import file_digest

__all__ = ["resolve_repaired", "expanded_folds", "GRAPH_LR_PROBE_NODES"]

GRAPH_LR_PROBE_NODES = (20161230, 20210827, 20260512)


def expanded_folds(dates, ends, ready, start=20170101):
    """1260/252 expanding windows, 63-day OOS, exact 11-session label purge."""
    dates, ends, ready = np.asarray(dates), np.asarray(ends), np.asarray(ready)
    if (
        not len(dates)
        or dates.shape != ends.shape
        or dates.shape != ready.shape
        or np.any(np.diff(dates) <= 0)
        or np.any(ends < dates)
    ):
        raise ValueError("Invalid repaired date or label endpoint arrays")
    mature = np.flatnonzero(ready & (ends <= dates[-1]))
    first = int(np.searchsorted(dates, start))
    if not len(mature) or first > mature[-1]:
        raise ValueError("No mature repaired labels for the requested OOS period")
    folds = []
    for a in range(first, int(mature[-1]) + 1, 63):
        v = a - 252
        begin = max(239, v - 1260)
        if v <= begin:
            raise ValueError("Insufficient prehistory; refusing to move OOS start")
        tr = np.arange(begin, v)
        va = np.arange(v, a)
        tr = tr[ends[tr] < dates[v]]
        va = va[ends[va] < dates[a]]
        te = np.arange(a, min(a + 63, int(mature[-1]) + 1))
        if not len(tr) or not len(va) or not ready[np.r_[tr, va, te]].all():
            raise ValueError(
                "Missing daily labels in repaired train/validation/test split"
            )
        folds.append(
            dict(
                train=dates[tr].astype(int).tolist(),
                valid=dates[va].astype(int).tolist(),
                test=dates[te].astype(int).tolist(),
                training_days_before_purge=v - begin,
                full_window=v - begin == 1260,
            )
        )
    return folds


def _latest_completed(root):
    candidates = []
    for summary in Path(root).glob("*/reports/summary.json"):
        report = json.loads(summary.read_text())
        marker = summary.with_name("dataset.json")
        if report.get("status") != "completed" or not marker.is_file():
            continue
        entry = json.loads(marker.read_text())
        candidates.append(
            (
                int(report.get("affected_end", report.get("end", 0))),
                summary.stat().st_mtime_ns,
                str(summary),
                entry["dataset"],
            )
        )
    if not candidates:
        raise ValueError(
            f"No completed repaired dataset with a dataset.json marker under {root}."
        )
    selected = max(candidates)
    return Path(selected[-1]), selected[-2]


def _validate_contract(manifest):
    if tuple(manifest["config"]["factors"]) != FACTORS:
        raise ValueError(
            "Repaired schedule requires the frozen 33-factor pool and directions"
        )
    expected = {
        "factorseq": [f"{c}__{f}" for c in STOCK_CHANNELS for f in FACTORS]
        + ["dtop__source_observed", "dtop__source_filled"],
        "factorperf": [
            f"{f}{suffix}" for f in PERF_NAMES for suffix in ("", "__valid")
        ],
        "factormarket": [
            f"{f}{suffix}" for f in MARKET_NAMES for suffix in ("", "__valid")
        ],
    }
    for key, names in expected.items():
        if manifest["blocks"][key]["features"] != names:
            raise ValueError(f"Unexpected repaired feature order: {key}")


def _label_state(dataset, dates):
    values = read_live_labels(dataset, dates)
    return np.asarray(np.isfinite(values).any(axis=0), dtype=bool), label_digest(values)


def resolve_repaired(source, model, name, mode):
    from src.proj import PATH, CALENDAR
    from src.res.algo.nn.model.factor_sequence.contextual_v4 import V4_VARIANTS
    from src.res.algo.nn.model.factor_sequence.contextual_v5 import V5_VARIANTS
    from src.res.model.experiment.factor_sequence.worklist import residual_config

    valid_modes = {
        "factorseq_residual": (
            "snapshot",
            "temporal",
            "contextual",
            "contextual_v4",
            "contextual_v4_market",
            "contextual_v4_events",
        ),
        "factorseq_contextual_v4": (*V4_VARIANTS, "graph_lr_probe"),
        "factorseq_contextual_v5": V5_VARIANTS,
    }
    if mode not in valid_modes.get(model, ()):
        raise ValueError(
            "Repaired schedules require a supported model and explicit mode"
        )
    pin_path = PATH.lc_share / "factor_sequence" / f"{source}.json"
    pinned = json.loads(pin_path.read_text()) if pin_path.exists() else None
    if not pinned and (model == "factorseq_contextual_v5" or mode == "graph_lr_probe"):
        raise FileNotFoundError(f"Existing Graph campaign pin required: {pin_path}")
    if pinned:
        dataset, source_report = Path(pinned["dataset"]), pinned["source_report"]
    else:
        dataset, source_report = _latest_completed(
            PATH.rslt_custom / "factor_sequence_fixes"
        )
    dataset = PATH.relocated_result(dataset, PATH.production, "factor_sequence_fixes")
    manifest = load_repaired_manifest(dataset)
    _validate_contract(manifest)
    manifest_hash = file_digest(dataset / "manifest.json")
    dates = np.load(dataset / "dates.npy")
    ends = np.asarray([int(CALENDAR.td(int(d), 11)) for d in dates])
    # Exclude immature tail before hashing, otherwise its later realization changes
    # the pin although the train/test labels have not changed.
    mature_dates = dates[ends <= dates[-1]]
    if not len(mature_dates):
        raise ValueError("No mature label dates")
    snapshot = None
    if pinned:
        if pinned["manifest_sha256"] != manifest_hash:
            raise ValueError(f"Frozen dataset manifest changed: expected {pinned['manifest_sha256']}, got {manifest_hash}")
        folds = pinned["folds"]
        if model == "factorseq_contextual_v5" or mode == "graph_lr_probe":
            graph = PATH.model_nn / "factorseq_contextual_v4@factorseq_v4_graph"
            snapshot = graph_label_snapshot(pin_path, pinned, dataset, mature_dates, graph)
        elif _label_state(dataset, mature_dates)[1] != pinned["label_sha256"]:
            raise ValueError("Frozen research inputs changed: historical label hash differs")
    else:
        ready, label_hash = _label_state(dataset, mature_dates)
        full_ready = np.zeros(len(dates), dtype=bool)
        full_ready[: len(ready)] = ready
        folds = expanded_folds(dates, ends, full_ready)
        for fold in folds:
            fold["model_date"] = int(CALENDAR.td(fold["test"][0], -1))
        pin = dict(
            dataset=str(dataset.resolve()),
            source_report=source_report,
            data_version=manifest["version"],
            manifest_sha256=manifest_hash,
            label_sha256=label_hash,
            folds=folds,
        )
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        staged = pin_path.with_name(f".{pin_path.name}.{uuid4().hex}.tmp")
        staged.write_text(json.dumps(pin, indent=2))
        try:
            # Publish a complete pin without overwriting a concurrent first task.
            os.link(staged, pin_path)
        except FileExistsError:
            return resolve_repaired(source, model, name, mode)
        finally:
            staged.unlink(missing_ok=True)
    config = residual_config(dataset, name, model, folds, mode, view="compact")
    if snapshot:
        config["input.factor_sequence.label_snapshot"] = snapshot
    return config
