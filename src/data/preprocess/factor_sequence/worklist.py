"""Resolve a checked, frozen server dataset when an ordinary schedule starts."""
from __future__ import annotations

__all__ = ['resolve_schedule']

import json
import re


def resolve_schedule(source, model, name, mode=None):
    from src.proj import PATH, CALENDAR
    from .bundle import load
    from .schedule_training import schedule_config, variants
    from pathlib import Path

    if not re.fullmatch(r"[A-Za-z0-9_-]+", source):
        raise ValueError("Invalid factor sequence source suite")
    if model not in dict(variants(False)) and model != "factorseq_residual":
        raise ValueError("Unsupported factor sequence model")
    suite = PATH.result / "factor_sequence" / source
    report = json.loads((suite / "suite.json").read_text())
    dataset = Path(report["dataset"])
    manifest = load(dataset)
    audit = json.loads((dataset / "audit.json").read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("version") != manifest["version"]
        or report.get("data_version") != manifest["version"]
    ):
        raise ValueError("Source suite requires a matching passed dataset audit")
    plan = json.loads((suite / "walk_forward.json").read_text())
    folds = [
        {**f, "model_date": int(CALENDAR.td(f["test"][0], -1))} for f in plan["folds"]
    ]
    if not folds:
        raise ValueError("Source suite has no historical folds")
    if model != "factorseq_residual":
        return schedule_config(dataset, name, model, folds, research=False, seed=17)
    if mode not in ("snapshot", "temporal", "contextual"):
        raise ValueError("Residual schedule requires an explicit mode")
    import numpy as np

    with np.load(dataset / "factorseq.npz", allow_pickle=False) as block:
        dates = block["dates"]
    ends = np.asarray([int(CALENDAR.td(int(d), 11)) for d in dates])
    expanded = []
    for old in folds:
        a = int(np.searchsorted(dates, old["test"][0]))
        v = a - 252
        start = max(239, v - 1260)
        tr = dates[start:v]
        va = dates[v:a]
        if v <= start or len(va) != 252:
            raise ValueError(
                "Insufficient history for expanding training/252-day validation"
            )
        tr = tr[ends[start:v] < va[0]]
        va = va[ends[v:a] < dates[a]]
        expanded.append(
            dict(
                train=tr.astype(int).tolist(),
                valid=va.astype(int).tolist(),
                test=old["test"],
                model_date=old["model_date"],
                training_days_before_purge=v - start,
                full_window=v - start == 1260,
            )
        )
    config = schedule_config(dataset, name, model, expanded, research=True, seed=17)
    keys = (
        ("factorseq", "factorperf", "factormarket")
        if mode == "contextual"
        else ("factorseq",)
    )
    config.update(
        {
            "model.window": 1800,
            "input.data.types": "+".join(keys),
            "input.factor_sequence.view": "compact",
            "input.sequence.lens": {
                k: (1 if mode == "snapshot" else 60) if k == "factorseq" else 120
                for k in keys
            },
            "input.sequence.steps": dict.fromkeys(keys, 1),
            "algo.factorseq_residual.mode": [mode],
        }
    )
    config["train.callbacks"].extend(
        ["DetailedAlphaAnalysis", "FactorResidualDiagnostics"]
    )
    config["callbacks.DetailedAlphaAnalysis"] = {"tasks": ["factor", "t50"]}
    return config
