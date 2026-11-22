"""Resolve a checked, frozen server dataset when an ordinary schedule starts."""

from __future__ import annotations

__all__ = ["resolve_schedule", "residual_config"]

import json
import re


def resolve_schedule(source, model, name, mode=None):
    from src.proj import PATH, CALENDAR
    from src.data.preprocess.factor_sequence.bundle import load
    from src.res.model.experiment.factor_sequence.schedule_training import (
        schedule_config,
        variants,
    )

    if not re.fullmatch(r"[A-Za-z0-9_-]+", source):
        raise ValueError("Invalid factor sequence source suite")
    if model not in dict(variants(False)) and model not in (
        "factorseq_residual",
        "factorseq_contextual_v4",
    ):
        raise ValueError("Unsupported factor sequence model")
    if source.startswith("repaired_"):
        from src.res.model.experiment.factor_sequence.repaired_schedule import (
            resolve_repaired,
        )

        return resolve_repaired(source, model, name, mode)
    if model == "factorseq_contextual_v4":
        raise ValueError("Contextual v4 requires a repaired frozen campaign")
    suite = PATH.rslt_custom / "factor_sequence" / source
    report = json.loads((suite / "suite.json").read_text())
    dataset = PATH.relocated_result(
        report["dataset"], PATH.production, "factor_sequence"
    )
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
    return residual_config(dataset, name, model, expanded, mode)


def residual_config(dataset, name, model, folds, mode, view="compact"):
    """Shared ordinary training policy; only the versioned input contract changes."""
    from src.res.model.experiment.factor_sequence.schedule_training import (
        schedule_config,
    )

    config = schedule_config(dataset, name, model, folds, research=True, seed=17)
    config["model.window"] = 1800
    config["input.factor_sequence.view"] = view
    config["train.callbacks"].extend(
        ["DetailedAlphaAnalysis", "FactorResidualDiagnostics"]
    )
    config["callbacks.DetailedAlphaAnalysis"] = {"tasks": ["factor", "t50"]}
    if model == "factorseq_contextual_v4":
        config.update(
            {
                "input.data.types": "factorseq+factorperf",
                "input.sequence.lens": {"factorseq": 60, "factorperf": 60},
                "input.sequence.steps": {"factorseq": 1, "factorperf": 1},
                "algo.factorseq_contextual_v4.variant": [mode],
            }
        )
        return config
    keys = ("factorseq",)
    v4_modes = ("contextual_v4", "contextual_v4_market", "contextual_v4_events")
    if mode in ("contextual", *v4_modes):
        keys += ("factorperf",)
    if mode in ("contextual", *v4_modes[1:]):
        keys += ("factormarket",)
    config.update(
        {
            "input.data.types": "+".join(keys),
            "input.sequence.lens": {
                k: (1 if mode == "snapshot" else 60)
                if k == "factorseq"
                else 1
                if k == "factormarket" and mode in v4_modes
                else 120
                for k in keys
            },
            "input.sequence.steps": dict.fromkeys(keys, 1),
            "algo.factorseq_residual.mode": [mode],
        }
    )
    if mode in v4_modes:
        config.update(
            {
                "algo.factorseq_residual.gate_hidden_dim": [8],
                "algo.factorseq_residual.gate_half_lives": [[5.0, 20.0, 60.0]],
                "algo.factorseq_residual.gate_scale": [0.2],
                "algo.factorseq_residual.gate_min_observations": [20],
                "algo.factorseq_residual.gate_penalty": [0.01],
            }
        )
    if mode in v4_modes[1:]:
        config["algo.factorseq_residual.market_rank"] = [2]
    if mode == "contextual_v4_events":
        config["algo.factorseq_residual.event_rank"] = [1]
        config["algo.factorseq_residual.event_decay_days"] = [20.0]
    return config
