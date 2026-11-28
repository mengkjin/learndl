"""Graph label migration, archived batch replay and frozen trainer input."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.preprocess.factor_sequence.bundle import file_digest
from src.res.model.experiment.factor_sequence import label_snapshot as labels
from src.res.model.util.data.module_data import ModuleData
from src.shared.numeric.tensor import standardize
from tests.test_factor_sequence_repaired_schedule import fixture

__all__ = ["GraphLabelSnapshotTests"]


class GraphLabelSnapshotTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.dataset, self.axis = fixture(root, n=12)
        self.dates = self.axis[:9]
        self.used = self.dates[[1, 2, 4, 5, 6]]
        self.folds = [
            dict(
                train=self.used[:2].tolist(),
                valid=self.used[2:3].tolist(),
                test=self.used[3:].tolist(),
            )
        ]
        self.live = np.tile(np.arange(9, dtype=np.float32), (2, 1))
        self.live[1] *= -0.5
        self.live[0, 5] = np.nan
        original = self.live.copy()
        original[:, 7:] = np.nan
        self.pin = dict(folds=self.folds, label_sha256=labels.label_digest(original))
        self.pin_path = root / "pin.json"
        self.graph = root / "graph"
        conf = self.graph / "configs"
        conf.mkdir(parents=True)
        (conf / "model.yaml").write_text(
            yaml.safe_dump(
                {
                    "model.name": "factorseq_v4_graph",
                    "model.labels": [labels.LABEL],
                    "input.factor_sequence.dataset": str(self.dataset),
                    "train.factor_sequence.folds": self.folds,
                }
            )
        )
        src = np.load(self.dataset / "factorseq.npy", mmap_mode="r+")
        src[:, :, 0, len(labels.FACTORS)] = 1.0
        src.flush()
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["factorseq.npy"] = file_digest(self.dataset / "factorseq.npy")
        manifest_path.write_text(json.dumps(manifest))
        self.archive = self.graph / "snapshot/factor_residual"
        self.archive.mkdir(parents=True)
        for i in (5, 6):
            col = torch.tensor(self.live[:, i])
            scores = standardize(standardize(col))
            mask = torch.isfinite(col)
            pd.DataFrame(
                dict(secid=np.array([1, 2])[mask.numpy()], label=scores[mask].numpy())
            ).to_feather(self.archive / f"{self.dates[i]}.feather")
        self.read = self.enterContext(
            patch.object(labels, "read_live_labels", return_value=self.live)
        )

    def freeze(self):
        return Path(
            labels.graph_label_snapshot(
                self.pin_path, self.pin, self.dataset, self.dates, self.graph
            )
        )

    def test_recover_reuse_and_real_label_loader(self):
        path = self.freeze()
        info = json.loads((path.parent / "verification.json").read_text())
        self.assertNotEqual(info["live_label_sha256"], info["legacy_label_sha256"])
        self.assertEqual(info["test_rows"], 3)
        self.assertEqual(info["cutoff"], self.dates[6])
        self.assertEqual(self.live[0, -1], 8)
        self.read.side_effect = AssertionError("live Y must not be read")
        self.assertEqual(self.freeze(), path)
        module = ModuleData.initialize(
            ["factorseq"],
            y_labels=[labels.LABEL],
            factor_sequence_dataset=str(self.dataset),
            factor_sequence_view="compact",
            factor_sequence_label_snapshot=str(path),
        )
        from src.proj import CALENDAR

        with (
            patch.object(
                module, "target_start_end", return_value=(self.dates[0], self.dates[-1])
            ),
            patch.object(CALENDAR, "range", return_value=self.dates),
            patch(
                "src.data.util.DataBlock.load_preprocess",
                side_effect=AssertionError("unexpected live Y read"),
            ),
        ):
            block = module.load().y
        np.testing.assert_array_equal(block.date, self.dates)
        np.testing.assert_equal(
            block.values[:, [1, 2, 4, 5, 6], 0, 0].numpy(),
            self.live[:, [1, 2, 4, 5, 6]],
        )
        self.assertTrue(
            np.isnan(block.align_date(self.dates).values[:, 7:, 0, 0].numpy()).all()
        )
        with path.open("ab") as stream:
            stream.write(b"broken")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            module.load_one("y", dates=self.dates)

    def test_changed_historical_value_and_mask_block_publication(self):
        for value in [5.0, np.nan]:
            with self.subTest(value=value):
                self.live[1, 2] = value
                with self.assertRaisesRegex(
                    ValueError, "Cannot reproduce Graph label pin"
                ):
                    self.freeze()
                self.assertFalse(list(self.pin_path.parent.glob("*.labels-v1.*")))

    def test_mismatch_in_actual_graph_output_blocks_publication(self):
        path = self.archive / f"{self.dates[6]}.feather"
        frame = pd.read_feather(path)
        frame["label"] += 0.1
        frame.to_feather(path)
        with self.assertRaisesRegex(ValueError, "Graph test batch labels differ"):
            self.freeze()
        frame.iloc[:1].to_feather(path)
        with self.assertRaisesRegex(ValueError, "Graph test security set changed"):
            self.freeze()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete Graph"):
            self.freeze()
