"""Existing-result comparisons and inference-only FiLM probes. Never fits a model."""

from __future__ import annotations
import json
from typing import Any, cast
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

__all__ = [
    'MODES',
    'paired_days',
    'film_probe',
    'inspect_delivery',
    'diagnose_roots',
    'archived_probes',
    'epoch_curves',
    'horizon_lags',
]

MODES = ("snapshot", "temporal", "contextual")


def paired_days(left, right):
    left, right = Path(left), Path(right)
    a = {p.name: p for p in left.glob("*.feather")}
    b = {p.name: p for p in right.glob("*.feather")}
    if not a or a.keys() != b.keys():
        raise ValueError("Missing predictions or unequal daily calendars")
    rows = []
    for name in sorted(a):
        x, y = pd.read_feather(a[name]), pd.read_feather(b[name])
        z = x.merge(y, on="secid", suffixes=("_a", "_b"), validate="one_to_one")
        if (
            len(z) != len(x)
            or len(z) != len(y)
            or not np.allclose(z.label_a, z.label_b, equal_nan=True)
        ):
            raise ValueError(f"Stock/label mismatch: {name}")
        for group, part in [
            ("all", z),
            ("low", z[z.coverage_a < 0.5]),
            ("medium", z[(z.coverage_a >= 0.5) & (z.coverage_a < 0.9)]),
            ("high", z[z.coverage_a >= 0.9]),
        ]:
            if len(part) < 2:
                continue
            ia = part.pred_a.corr(part.label_a, method="spearman")
            ib = part.pred_b.corr(part.label_b, method="spearman")
            rows.append(
                dict(
                    date=int(Path(name).stem),
                    group=group,
                    n=len(part),
                    base_ic=ia,
                    candidate_ic=ib,
                    delta_ic=ib - ia,
                    prediction_corr=part.pred_a.corr(part.pred_b, method="spearman"),
                    linear_std=part.linear_b.std(),
                    residual_std=part.residual_b.std(),
                    branch_corr=part.linear_b.corr(part.residual_b),
                )
            )
    return pd.DataFrame(rows)


def film_probe(model, streams, labels=None, event_indices=()):
    """Real loaded model/batch only. Temporal perturbation holds the first state
    throughout the window (no shuffled days or future data). No optimizer step.
    Gradients are predictive-loss gradients when labels supplied, otherwise
    output sensitivity; the two must not be confused.
    """
    import torch

    if model.mode != "contextual":
        raise ValueError("Contextual model required")
    old = model.training
    model.eval()
    x = [v.detach().clone() for v in streams]
    capture = {}
    handles = [
        model.film.register_forward_hook(lambda m, i, o: capture.update(film=o)),
        model.project.register_forward_hook(lambda m, i, o: capture.update(project=o)),
    ]
    result = {}
    try:
        with torch.enable_grad():
            pred, other = model(x)
            loss = (
                pred.square().mean()
                if labels is None
                else (pred.flatten() - labels.flatten()).square().mean()
            )
            params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
            result["gradient_kind"] = (
                "output_energy_sensitivity" if labels is None else "mse"
            )
            result["gradient_norms"] = {
                n: None if g is None else float(g.norm())
                for (n, _), g in zip(params, grads)
            }
        gamma, beta = capture["film"].detach().chunk(2, -1)
        z = capture["project"].detach()
        gamma, beta = gamma[:, -z.shape[1] :], beta[:, -z.shape[1] :]
        for name, v in [("gamma", gamma), ("beta", beta)]:
            result[name] = dict(
                mean=float(v.mean()),
                std=float(v.std(unbiased=False)),
                max_abs=float(v.abs().max()),
                temporal_change=float(v.diff(dim=1).abs().mean()),
            )
        result["relative_modulation_rms"] = float(
            (z * gamma + beta).square().mean().sqrt()
            / z.square().mean().sqrt().clamp_min(1e-12)
        )
        result["context_parameters"] = sum(
            p.numel()
            for n, p in params
            if n.startswith(("performance.", "market.", "film."))
        )
        result["context_explicit_dropout"] = False
        baseline = pred.detach()
        with torch.no_grad():
            handle = model.film.register_forward_hook(
                lambda m, i, o: torch.zeros_like(o)
            )
            try:
                result["disable_film_rms"] = float(
                    (model(x)[0] - baseline).square().mean().sqrt()
                )
            finally:
                handle.remove()
            for name, idx in [("performance", 1), ("market", 2), ("events", 2)]:
                modified = [v.clone() for v in x]
                if name == "events":
                    if not event_indices:
                        result["events"] = (
                            "unavailable: explicit event channel indices required"
                        )
                        continue
                    modified[idx][..., list(event_indices)] = modified[idx][
                        :, :1, ..., list(event_indices)
                    ]
                else:
                    modified[idx] = (
                        modified[idx][:, :1].expand_as(modified[idx]).clone()
                    )
                result[name + "_held_first_rms"] = float(
                    (model(modified)[0] - baseline).square().mean().sqrt()
                )
        result["branch_std"] = {
            n: float(other[n].detach().std(unbiased=False))
            for n in ("linear_score", "residual_score")
        }
    finally:
        for h in handles:
            h.remove()
        model.train(old)
    return result


def inspect_delivery(roots):
    from src.automation.artifact_exchange.core import Config

    c = Config.load()
    records = []
    names = {Path(p).name for p in roots.values()}
    for directory in (c.state / "outbox", c.state / "received", c.results):
        if not directory.exists():
            continue
        pattern = "source.json" if directory == c.results else "*.json"
        for p in directory.rglob(pattern):
            try:
                text = p.read_text()
                if any(n in text for n in names):
                    records.append(dict(path=str(p), record=json.loads(text)))
            except (OSError, ValueError):
                continue
    return dict(
        enabled=c.enabled,
        role=c.role,
        shared=str(c.shared_root),
        state=str(c.state),
        records=records,
        shared_packets=[
            str(p)
            for p in (c.shared_root / "artifact_exchange").rglob("*.transfer.zip")
            if any(n in str(p) for n in names)
        ],
        note="Only published records prove local publication. Receiver receipts are local to the receiver; absence here does not prove delivery failure.",
    )


def diagnose_roots(roots, folder):
    """Explicit actual roots including @2; never substitute another experiment."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = dict(
        status="incomplete", models={}, comparisons=[], pending=[]
    )
    configs = {}
    for mode in MODES:
        root = Path(roots[mode]) if mode in roots else None
        if root is None or not root.is_dir():
            result["pending"].append(f"{mode}: exact model root unavailable: {root}")
            continue
        path = root / "configs/model.yaml"
        try:
            config = yaml.safe_load(path.read_text())
            configs[mode] = config
            archives = sorted(
                int(p.name)
                for p in (root / "archive/0").iterdir()
                if p.is_dir() and p.name.isdigit()
            )
            folds = config.get("train.factor_sequence.folds", [])
            planned = [int(f["model_date"]) for f in folds]
            info = dict(
                root=str(root.resolve()),
                archive_dates=archives,
                archive_count=len(archives),
                configured_interval=config.get("model.interval"),
                frozen_fold_count=len(folds),
                missing_frozen_dates=sorted(set(planned) - set(archives)),
                selected_probe_checkpoints=[
                    archives[i]
                    for i in sorted(set([0, len(archives) // 2, len(archives) - 1]))
                ]
                if archives
                else [],
                effective_dates=[
                    dict(
                        model_date=f["model_date"],
                        train_dates=len(f["train"]),
                        valid_dates=len(f["valid"]),
                        test_dates=len(f["test"]),
                    )
                    for f in folds
                ],
                optimizer_configuration={
                    k: v
                    for k, v in config.items()
                    if any(s in k.lower() for s in ("optim", "weight_decay"))
                },
                results=[
                    str(p.relative_to(root))
                    for p in (root / "results").rglob("*")
                    if p.is_file()
                ],
            )
            lifecycle = root / "configs/lifecycle.yaml"
            info["lifecycle"] = (
                yaml.safe_load(lifecycle.read_text()) if lifecycle.exists() else {}
            )
            node_manifest = root / "snapshot/model_manifest.json"
            info["node_manifest"] = (
                json.loads(node_manifest.read_text()) if node_manifest.exists() else {}
            )
            budget = info["lifecycle"].get("budget", {})
            step = (
                budget.get("step", 2)
                if info["lifecycle"].get("stage") == "exploration"
                and budget.get("enabled")
                else 1
            )
            info["lifecycle_selected_dates"] = planned[::step]
            info["archive_matches_lifecycle_grid"] = archives == planned[::step]
            info["distinct_context_training_dates"] = [
                len(set(f["train"])) for f in folds
            ]
            curve_output = folder / mode
            curve_output.mkdir(exist_ok=True)
            try:
                info["epoch_curves"] = epoch_curves(root, curve_output)
            except Exception as exc:
                result["pending"].append(f"{mode} epoch curves: {exc}")
            result["models"][mode] = info
            # Preserve exact configuration used in diagnostics, not current schedule defaults.
            (folder / (mode + "_model.yaml")).write_text(path.read_text())
        except Exception as exc:
            result["pending"].append(f"{mode}: {exc}")
    for base, candidate in [("snapshot", "temporal"), ("temporal", "contextual")]:
        if base not in configs or candidate not in configs:
            continue
        try:
            keys = (
                "input.factor_sequence.dataset",
                "train.factor_sequence.folds",
                "model.labels",
            )
            for key in keys:
                if configs[base].get(key) != configs[candidate].get(key):
                    raise ValueError(f"Unmatched provenance: {key}")
            if not configs[base].get("input.factor_sequence.dataset"):
                raise ValueError("Missing dataset identity")
            pair = candidate + "-" + base
            daily = paired_days(
                Path(roots[base]) / "results/factor_residual",
                Path(roots[candidate]) / "results/factor_residual",
            )
            daily.to_csv(folder / (pair + "_daily.csv"), index=False)
            daily.assign(year=daily.date // 10000).groupby(["year", "group"]).mean(
                numeric_only=True
            ).to_csv(folder / (pair + "_annual.csv"))
            result["comparisons"].append(pair)
        except Exception as exc:
            result["pending"].append(f"{candidate}-{base}: {exc}")
    result["delivery"] = inspect_delivery(roots)
    if "contextual" in configs:
        try:
            result["weight_probes"] = archived_probes(
                roots["contextual"], configs["contextual"], folder
            )
        except Exception as exc:
            result["pending"].append(f"Archived weight/input probes: {exc}")
    result["pending"].append(
        "Effective optimizer param-group state and stop reasons must be read from retained checkpoint/training logs; configured weight decay is not proof of optimizer state."
    )
    result["recommendations"] = [
        dict(
            priority=1,
            change="smaller context encoder / bounded FiLM with explicit regularization",
            evidence="Context GRUs have no explicit dropout and FiLM has unbounded gamma/beta. Confirm with real probes before selecting.",
        ),
        dict(
            priority=2,
            change="factor-aligned performance encoding",
            evidence="Flattened context input retains channel identity via weights but imposes no correspondence to stock factor contributions.",
        ),
        dict(
            priority=3,
            change="evaluate daily-performance / ten-day-target mismatch",
            evidence="Requires fixed lag report; no best-horizon selection in this audit.",
        ),
    ]
    (folder / "diagnostics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str)
    )
    return dict(
        status=result["status"],
        pending=result["pending"],
        report=str(folder / "diagnostics.json"),
    )


def archived_probes(root, config, output, max_stocks=256):
    """Disk-backed extraction prevents NPZ decompression from allocating the full
    stock history in RAM. Uses three fixed checkpoints, first OOS day per fold,
    and recorded stock membership. Never invokes the training DataModule.
    """
    import shutil
    import tempfile
    from zipfile import ZipFile
    import torch
    from src.data.preprocess.factor_sequence.bundle import load as load_manifest
    from src.data.preprocess.factor_sequence.schema import EVENTS
    from src.res.algo.nn.model.factor_sequence.residual import FactorResidual

    root, output = Path(root), Path(output)
    dataset = Path(config["input.factor_sequence.dataset"])
    manifest = load_manifest(dataset)
    factors = manifest["config"]["factors"]
    archives = sorted(
        int(p.name) for p in (root / "archive/0").iterdir() if p.name.isdigit()
    )
    selected = [archives[i] for i in sorted({0, len(archives) // 2, len(archives) - 1})]
    folds = {int(f["model_date"]): f for f in config["train.factor_sequence.folds"]}
    days = [int(folds[a]["test"][0]) for a in selected]
    records = [
        pd.read_feather(root / "results/factor_residual" / f"{d}.feather")
        .sort_values("secid")
        .head(max_stocks)
        for d in days
    ]
    streams = [[] for _ in days]
    market_names = []
    for key, length in [("factorseq", 60), ("factorperf", 120), ("factormarket", 120)]:
        with np.load(dataset / f"{key}.npz", allow_pickle=False) as meta:
            names = list(meta["features"])
            dates = meta["dates"]
            secids = meta["secids"]
        if names != manifest["blocks"][key]["features"]:
            raise ValueError("Feature order differs from manifest")
        if key == "factorseq":
            wanted = [
                f"{c}__{f}" for c in ("value", "valid", "gap", "never") for f in factors
            ]
        elif key == "factorperf":
            wanted = [
                f"{f}__{m}{s}"
                for f in factors
                for m in (
                    "ic",
                    "spread",
                    "ic_mean20",
                    "ic_mean60",
                    "spread_mean20",
                    "spread_mean60",
                )
                for s in ("", "__valid")
            ]
        else:
            wanted = names
            market_names = names
        indices = [names.index(n) for n in wanted]
        with tempfile.TemporaryDirectory(prefix="factor-audit-", dir=output) as tmp:
            with ZipFile(dataset / f"{key}.npz") as z:
                info = z.getinfo("values.npy")
                if shutil.disk_usage(tmp).free < info.file_size * 1.1:
                    raise RuntimeError(
                        f"Insufficient scratch disk for bounded-memory {key} probe"
                    )
                target = Path(tmp) / "values.npy"
                with z.open(info) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest, 1024 * 1024)
            mapped = np.load(target, mmap_mode="r", allow_pickle=False)
            for i, d in enumerate(days):
                match = np.flatnonzero(dates == d)
                if len(match) != 1 or match[0] + 1 < length:
                    raise ValueError(f"Incomplete context window: {key} {d}")
                stop = int(match[0]) + 1
                ids = (
                    pd.Index(secids).get_indexer(pd.Index(records[i].secid))
                    if key == "factorseq"
                    else np.array([0])
                )
                if (ids < 0).any():
                    raise ValueError("Recorded stock absent from immutable dataset")
                # Slice time before advanced row indexing to bound RAM.
                small = np.array(
                    mapped[:, stop - length : stop, 0, :][ids][..., indices], copy=True
                )
                if not np.isfinite(small).all():
                    raise ValueError("Nonfinite archived model inputs")
                streams[i].append(torch.from_numpy(small).float())
            del mapped
    results = []
    for checkpoint, day, x, record in zip(selected, days, streams, records):
        path = root / "archive/0" / str(checkpoint) / "best/state_dict.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        dims = [v.shape[-1] for v in x]
        hidden = state["project.weight"].shape[0]
        layers = sum(k.startswith("stock.weight_ih_l") for k in state)
        clayers = sum(k.startswith("performance.weight_ih_l") for k in state)
        model = FactorResidual(
            dims,
            mode="contextual",
            hidden_dim=hidden,
            rnn_layers=layers,
            context_layers=clayers,
        )
        model.load_state_dict(state, strict=True)
        # Existing production inputs only, never the new slim candidate view.
        events = [
            j
            for j, n in enumerate(market_names)
            if any(n.startswith(e) for e in EVENTS)
        ]
        model.eval()
        with torch.no_grad():
            prediction = model(x)[0].flatten().numpy()
        if not np.allclose(prediction, record.pred.to_numpy(), rtol=1e-4, atol=1e-5):
            raise ValueError(
                f"Cannot reconstruct archived predictions at {day}; sensitivity results not accepted"
            )
        probe = film_probe(
            model, x, torch.tensor(record.label.to_numpy(), dtype=torch.float32), events
        )
        probe.update(
            checkpoint=checkpoint,
            date=day,
            stocks=len(record),
            reconstruction_max_error=float(
                np.max(np.abs(prediction - record.pred.to_numpy()))
            ),
            input_data_version=manifest["version"],
        )
        results.append(probe)
        (output / "weight_probes.json").write_text(json.dumps(results, indent=2))
    (output / "weight_probes.json").write_text(json.dumps(results, indent=2))
    return results


def epoch_curves(root, output):
    """Read saved per-epoch tables, preserving retries and phases."""
    frames = []
    summaries = []
    for path in sorted((Path(root) / "snapshot/metrics").glob("model.*.xlsx")):
        with pd.ExcelFile(path) as book:
            for dataset in ("train", "valid"):
                name = dataset + "_epoch_totals"
                if name not in book.sheet_names:
                    continue
                frame = pd.read_excel(book, sheet_name=name)
                frame["dataset"] = dataset
                frame["checkpoint"] = int(path.stem.split(".")[-1])
                frames.append(frame)
                if (
                    dataset == "valid"
                    and "rankic" in frame
                    and frame.rankic.notna().any()
                ):
                    best = frame.loc[frame.rankic.idxmax()]
                    summaries.append(
                        dict(
                            checkpoint=int(path.stem.split(".")[-1]),
                            diagnostic_best_valid_rankic=float(cast(Any, best.rankic)),
                            diagnostic_best_epoch=best.get("epoch"),
                            epochs_recorded=len(frame),
                            note="Rank IC maximum across saved attempts; not necessarily the selected archived attempt",
                        )
                    )
    if not frames:
        raise FileNotFoundError("Saved per-epoch metric workbooks absent")
    pd.concat(frames, ignore_index=True).to_csv(
        output / "epoch_curves.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(output / "epoch_summary.csv", index=False)
    return dict(
        workbooks=len({x["checkpoint"].iloc[0] for x in frames}),
        report=str(output / "epoch_curves.csv"),
    )


def horizon_lags(performance, outcomes, output):
    """Fixed-lag descriptive correlations on realized archived labels; no tuning.
    Outcomes are per-factor cross-sectional IC with the saved ten-day label.
    Daily factor values are current source snapshots (revision caveat retained).
    """
    rows = []
    for name in outcomes:
        for metric in ("ic", "spread"):
            feature = name + "__" + metric
            for lag in (0, 1, 5, 10, 20, 60):
                pair = pd.concat(
                    [
                        performance[feature].shift(lag).rename("past"),
                        outcomes[name].rename("target"),
                    ],
                    axis=1,
                ).dropna()
                rows.append(
                    dict(
                        factor=name,
                        input=metric,
                        lag_trading_days=lag,
                        n=len(pair),
                        correlation=pair.past.corr(pair.target)
                        if len(pair) > 2
                        else np.nan,
                    )
                )
    pd.DataFrame(rows).to_csv(output, index=False)
