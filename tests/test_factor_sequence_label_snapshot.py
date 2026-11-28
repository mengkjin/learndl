"""Current-label freezing and reuse through the ordinary data loader."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.res.model.experiment.factor_sequence import label_snapshot as labels
from src.res.model.util.data.module_data import ModuleData
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
        self.pin_path = root / "pin.json"
        self.read = self.enterContext(
            patch.object(
                labels,
                "read_live_labels",
                side_effect=lambda dataset, dates: self.live[
                    :, np.searchsorted(self.dates, dates)
                ],
            )
        )

    def freeze(self):
        return Path(
            labels.current_label_snapshot(
                self.pin_path, "bundle-sha", self.folds, self.dataset, self.dates
            )
        )

    def test_recover_reuse_and_real_label_loader(self):
        path = self.freeze()
        info = json.loads((path.parent / "verification.json").read_text())
        self.assertEqual(info["source"], "fit/Y")
        self.assertEqual(info["fold_days"], len(self.used))
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

    def test_first_snapshot_uses_current_values_and_masks(self):
        self.live[1, 2] = np.nan
        self.live[0, 2] = 37.0
        path = self.freeze()
        self.live[:, 2] = -100.0
        self.assertEqual(path, self.freeze())
        block = labels.load_label_snapshot(path)
        np.testing.assert_equal(block.values[:, 1, 0, 0].numpy(), [37.0, np.nan])

    def test_missing_entire_fold_date_blocks_publication(self):
        self.live[:, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing labels"):
            self.freeze()
        self.assertFalse(list(self.pin_path.parent.glob("*.v5-current-labels.*")))
