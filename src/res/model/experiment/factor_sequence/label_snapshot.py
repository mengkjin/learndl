"""Freeze the current fit labels once for the V5 schedules and learning-rate probe."""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.data.preprocess.factor_sequence.bundle import file_digest, write_json

__all__ = [
    "LABEL",
    "read_live_labels",
    "label_digest",
    "current_label_snapshot",
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


def label_digest(values):
    """Reproduce the original stock-major little-endian float32 pin without editing Y."""
    digest = hashlib.sha256()
    for offset in range(0, len(values), 256):
        chunk = values[offset : offset + 256].astype("<f4", copy=True)
        chunk[np.isnan(chunk)] = np.nan
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def current_label_snapshot(pin_path, manifest_hash, folds, dataset, mature_dates):
    """Share one immutable current-label snapshot across the three schedules."""
    identity = hashlib.sha256(
        json.dumps(
            {"manifest": manifest_hash, "folds": folds, "label": LABEL},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    folder = pin_path.parent / f"{pin_path.stem}.v5-current-labels.{identity}"
    receipt = folder / "verification.json"
    if not folder.exists():
        dates = np.unique(
            [d for f in folds for split in ("train", "valid", "test") for d in f[split]]
        )
        if not dates.size or not np.isin(dates, mature_dates).all():
            raise ValueError("Frozen fold dates outside mature label grid")
        secids = np.load(dataset / "secids.npy")
        values = np.ascontiguousarray(read_live_labels(dataset, dates), dtype="<f4")
        if not np.isfinite(values).any(axis=0).all():
            raise ValueError("Missing labels on a pinned fold date")
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
                    campaign_sha256=identity,
                    snapshot_sha256=checksum,
                    source="fit/Y",
                    cutoff=int(dates[-1]),
                    fold_days=len(dates),
                    securities=len(secids),
                ),
            )
            try:
                stage.rename(folder)
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
    info = json.loads(receipt.read_text())
    if info["campaign_sha256"] != identity:
        raise ValueError(f"Wrong verification receipt: {receipt}")
    path = folder / f"{info['snapshot_sha256']}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    from src.proj import Logger

    Logger.info(
        f"Frozen current fit/Y labels through {info['cutoff']}, {info['fold_days']} fold dates, "
        f"{info['securities']} securities. Receipt: {receipt}"
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
