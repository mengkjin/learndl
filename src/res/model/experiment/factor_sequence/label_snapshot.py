"""Verify legacy Graph labels and publish the fold union for ordinary training."""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.preprocess.factor_sequence.bundle import file_digest, write_json
from src.data.preprocess.factor_sequence.schema import FACTORS, STOCK_CHANNELS
from src.shared.numeric.tensor import standardize

__all__ = [
    "LABEL",
    "read_live_labels",
    "label_digest",
    "graph_label_snapshot",
    "load_label_snapshot",
]

LABEL = "std_lag1_10"


def read_live_labels(dataset, dates):
    from src.data.util import DataBlock

    with DataBlock.preprocess_lock("y", "fit"):
        block = DataBlock.load_preprocess("y", frame="fit")
    if not block.initiated or LABEL not in block.feature:
        raise ValueError(f"Missing fit label {LABEL}")
    block = (
        block.align_feature([LABEL], inplace=True)
        .align_secid(np.load(dataset / "secids.npy"), inplace=True)
        .align_date(dates, inplace=True)
    )
    return block.values[:, :, 0, 0].detach().cpu().numpy()


def label_digest(values, *, nan_from=None):
    """Reproduce the original stock-major little-endian float32 pin without editing Y."""
    digest = hashlib.sha256()
    for offset in range(0, len(values), 256):
        chunk = values[offset : offset + 256].astype("<f4", copy=True)
        if nan_from is not None:
            chunk[:, nan_from:] = np.nan
        chunk[np.isnan(chunk)] = np.nan
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _check_graph(graph, dataset, folds, dates, secids, values):
    """Check the saved Graph configuration and each actual OOS batch's securities and label."""
    from src.res.model.analytic.factor_sequence.context_diagnostics import (
        residual_snapshot,
    )

    config = yaml.safe_load((graph / "configs/model.yaml").read_text())
    if (
        config["model.name"] != "factorseq_v4_graph"
        or config["model.labels"] != [LABEL]
        or Path(config["input.factor_sequence.dataset"]) != dataset
        or config["train.factor_sequence.folds"] != folds
    ):
        raise ValueError(f"Graph benchmark dataset, target or folds differ: {graph}")
    days = sorted(day for f in folds for day in f["test"])
    folder = residual_snapshot(graph)
    if sorted(int(p.stem) for p in folder.glob("*.feather")) != days:
        raise ValueError(f"Incomplete Graph test label archive: {folder}")

    # The compact view selects the full bundle's endpoint validity channels.
    stock = np.load(dataset / "factorseq.npy", mmap_mode="r")
    axis = np.load(dataset / "dates.npy")
    begin = np.searchsorted(axis, days[0])
    end = np.searchsorted(axis, days[-1]) + 1
    k = len(FACTORS)
    feature = STOCK_CHANNELS.index("valid") * k
    valid = np.empty((len(secids), end - begin), dtype=bool)
    for offset in range(0, len(secids), 32):
        part = stock[offset : offset + 32, begin:end, 0, feature : feature + k]
        valid[offset : offset + 32] = (part > 0).any(axis=-1)

    largest_error = 0.0
    rows = 0
    for day in days:
        frame = pd.read_feather(
            folder / f"{day}.feather", columns=["secid", "label"]
        ).set_index("secid")
        column = values[:, np.searchsorted(dates, day)]
        effective = valid[:, np.searchsorted(axis, day) - begin] & np.isfinite(column)
        ids = secids[effective]
        if (
            not len(ids)
            or not frame.index.is_unique
            or len(frame) != len(ids)
            or not np.isin(ids, frame.index).all()
        ):
            raise ValueError(f"Graph test security set changed on {day}")
        # DataModule standardizes first on the full cross-section, then on the effective batch.
        score = standardize(torch.from_numpy(column.copy()), dim=0)
        score[~torch.from_numpy(effective)] = torch.nan
        expected = standardize(score, dim=0).numpy()[effective]
        actual = frame["label"].reindex(ids).to_numpy()
        # Only this replay of floating-point reductions allows rounding error.
        # The separately verified pre-standardization input hash is bit-exact.
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Graph test batch labels differ on {day}: max abs error {np.max(np.abs(actual - expected))}"
            )
        largest_error = max(largest_error, float(np.max(np.abs(actual - expected))))
        rows += len(ids)
    return dict(
        graph=str(graph),
        test_days=len(days),
        test_rows=rows,
        max_abs_batch_error=largest_error,
    )


def graph_label_snapshot(pin_path, pin, dataset, mature_dates, graph):
    """Publish a verified snapshot once per immutable pin, including simultaneous first users."""
    identity = hashlib.sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest()
    folder = pin_path.parent / f"{pin_path.stem}.labels-v1.{identity}"
    receipt = folder / "verification.json"
    if not folder.exists():
        folds = pin["folds"]
        dates = np.unique(
            [d for f in folds for split in ("train", "valid", "test") for d in f[split]]
        )
        if not dates.size or not np.isin(dates, mature_dates).all():
            raise ValueError("Frozen fold dates outside mature label grid")
        secids = np.load(dataset / "secids.npy")
        live = read_live_labels(dataset, mature_dates)
        cutoff = int(dates[-1])
        full = label_digest(live)
        replay = full
        if full != pin["label_sha256"]:
            replay = label_digest(
                live, nan_from=np.searchsorted(mature_dates, cutoff, side="right")
            )
        if replay != pin["label_sha256"]:
            raise ValueError(
                f"Cannot reproduce Graph label pin through {cutoff}. Expected {pin['label_sha256']}, "
                f"live {full}, restored NaN tail {replay}. Original fit/Y required."
            )
        values = np.ascontiguousarray(
            live[:, np.searchsorted(mature_dates, dates)], dtype="<f4"
        )
        del live
        if not np.isfinite(values).any(axis=0).all():
            raise ValueError("Missing labels on a pinned fold date")
        checked = _check_graph(graph, dataset, folds, dates, secids, values)
        with TemporaryDirectory(dir=pin_path.parent) as tmp:
            stage = Path(tmp) / "snapshot"
            stage.mkdir()
            payload = stage / "labels.npz"
            np.savez(payload, values=values, dates=dates, secids=secids, label=LABEL)
            checksum = file_digest(payload)
            payload.rename(stage / f"{checksum}.npz")
            write_json(
                stage / receipt.name,
                dict(
                    pin_sha256=identity,
                    snapshot_sha256=checksum,
                    legacy_label_sha256=replay,
                    live_label_sha256=full,
                    cutoff=cutoff,
                    fold_days=len(dates),
                    securities=len(secids),
                    **checked,
                ),
            )
            try:
                stage.rename(folder)
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
    info = json.loads(receipt.read_text())
    if (
        info["pin_sha256"] != identity
        or info["legacy_label_sha256"] != pin["label_sha256"]
    ):
        raise ValueError(f"Wrong verification receipt: {receipt}")
    path = folder / f"{info['snapshot_sha256']}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    from src.proj import Logger

    Logger.info(
        f"Verified Graph labels through {info['cutoff']}, {info['fold_days']} fold dates, "
        f"{info['test_rows']} archived test labels. Receipt: {receipt}"
    )
    return str(path)


def load_label_snapshot(path):
    from src.data.util import DataBlock

    path = Path(path)
    if file_digest(path) != path.stem:
        raise ValueError(f"Corrupt frozen Graph labels: {path}")
    with np.load(path, allow_pickle=False) as snapshot:
        values, dates, secids = (snapshot[k] for k in ("values", "dates", "secids"))
        if (
            snapshot["label"].item() != LABEL
            or values.shape != (len(secids), len(dates))
            or values.dtype != np.dtype("<f4")
            or np.any(np.diff(dates) <= 0)
            or len(np.unique(secids)) != len(secids)
        ):
            raise ValueError(f"Invalid frozen Graph label schema: {path}")
    return DataBlock(
        values=torch.from_numpy(values[:, :, None, None]),
        date=dates,
        secid=secids,
        feature=np.array([LABEL]),
    )
