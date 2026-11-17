"""Repaired data selection, shared campaign freezing and ordinary schedule inputs."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.preprocess.factor_sequence.schema import FACTORS, STOCK_CHANNELS
from src.data.preprocess.factor_sequence.context_audit import PERF_NAMES, MARKET_NAMES
from src.data.preprocess.factor_sequence.bundle import file_digest
from src.data.preprocess.factor_sequence.rebuild_data import REPAIR_VERSION
from src.data.preprocess.factor_sequence.repaired_schedule import expanded_folds
from src.data.preprocess.factor_sequence.worklist import resolve_schedule
from src.data.preprocess.factor_sequence.processors import load_block
from src.res.algo.nn.model.factor_sequence.residual import FactorResidual

__all__ = ["RepairedScheduleTests"]


def fixture(root, run="first", n=1900, status="completed"):
    folder = root / "factor_sequence_fixes" / run / "reports"
    folder.mkdir(parents=True)
    dataset = root / ("dataset_" + run)
    dataset.mkdir()
    dates = (
        pd.bdate_range("2011-01-03", periods=n + 30)
        .strftime("%Y%m%d")
        .astype(int)
        .to_numpy()
    )
    np.save(dataset / "dates.npy", dates[:n])
    np.save(dataset / "secids.npy", np.array([1, 2]))
    features = {
        "factorseq": [f"{c}__{f}" for c in STOCK_CHANNELS for f in FACTORS]
        + ["dtop__source_observed", "dtop__source_filled"],
        "factorperf": [
            f"{name}{suffix}" for name in PERF_NAMES for suffix in ("", "__valid")
        ],
        "factormarket": [
            f"{name}{suffix}" for name in MARKET_NAMES for suffix in ("", "__valid")
        ],
    }
    for key, names in features.items():
        np.save(
            dataset / f"{key}.npy",
            np.zeros(
                (2 if key == "factorseq" else 1, n, 1, len(names)), dtype="float32"
            ),
        )
    manifest = dict(
        format=REPAIR_VERSION,
        version=run,
        config={"factors": list(FACTORS)},
        files={p.name: file_digest(p) for p in dataset.glob("*.npy")},
        blocks={k: dict(features=v) for k, v in features.items()},
    )
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (folder / "summary.json").write_text(
        json.dumps(dict(status=status, affected_end=int(dates[n - 1])))
    )
    (folder / "dataset.json").write_text(json.dumps(dict(dataset=str(dataset))))
    return dataset, dates


class RepairedScheduleTests(unittest.TestCase):
    def test_three_schedules_pin_same_data_and_use_compact_inputs(self):
        from src.proj import PATH, CALENDAR

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            dataset, dates = fixture(root)
            stack.enter_context(patch.object(PATH, "result", root))
            stack.enter_context(patch.object(PATH, "lc_share", root / "shared"))
            stack.enter_context(
                patch.object(
                    CALENDAR,
                    "td",
                    side_effect=lambda d, n: int(dates[np.searchsorted(dates, d) + n]),
                )
            )
            stack.enter_context(
                patch(
                    "src.data.preprocess.factor_sequence.repaired_schedule._label_state",
                    side_effect=lambda _, ds: (np.ones(len(ds), bool), "stable_labels"),
                )
            )
            for mode in ("snapshot", "temporal", "contextual"):
                config = resolve_schedule(
                    "repaired_20261117", "factorseq_residual", "test_" + mode, mode
                )
                self.assertEqual(
                    config["input.factor_sequence.dataset"], str(dataset.resolve())
                )
                self.assertEqual(config["input.factor_sequence.view"], "compact")
                self.assertEqual(
                    config["callbacks.DetailedAlphaAnalysis"],
                    {"tasks": ["factor", "t50"]},
                )
                self.assertEqual(config["train.max_epoch"], 30)
                self.assertEqual(config["callbacks.EarlyStoppage"]["peak_patience"], 5)
                self.assertEqual(config["model.interval"], 63)
                from src.res.model.util.config.config import ScheduleConfig

                effective = ScheduleConfig(schedule_name=f"factorseq_v3_{mode}")
                self.assertEqual(
                    effective["input.factor_sequence.dataset"], str(dataset.resolve())
                )
                self.assertEqual(effective["algo.factorseq_residual.mode"], [mode])
                self.assertEqual(effective["model.name"], f"factorseq_v3_{mode}")
                self.assertEqual(config["env.random_seed"], 17)
                keys = config["input.data.types"].split("+")
                self.assertEqual(len(keys), 3 if mode == "contextual" else 1)
                lens = config["input.sequence.lens"]
                inputs = [
                    load_block(
                        dataset, k, dates=dates[1600 - lens[k] : 1600], view="compact"
                    ).values.squeeze(2)
                    for k in keys
                ]
                widths = [x.shape[-1] for x in inputs]
                self.assertEqual(
                    widths, [132, 132, 30] if mode == "contextual" else [132]
                )
                network = FactorResidual(widths, mode=mode)
                score, _ = network(inputs)
                self.assertEqual(score.shape, (2, 1))
                self.assertTrue(torch.isfinite(score).all())
                if mode == "snapshot":
                    fixture(root, run="newer", n=1910)
            pin = json.loads(
                (
                    root / "shared" / "factor_sequence" / "repaired_20261117.json"
                ).read_text()
            )
            self.assertEqual(pin["data_version"], "first")
            self.assertEqual(pin["folds"][0]["test"][0], 20170102)
            with patch(
                "src.data.preprocess.factor_sequence.repaired_schedule._label_state",
                side_effect=lambda _, ds: (np.ones(len(ds), bool), "revised"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "Frozen research inputs changed"
                ):
                    resolve_schedule(
                        "repaired_20261117",
                        "factorseq_residual",
                        "test_snapshot",
                        "snapshot",
                    )

    def test_no_completed_dataset_never_falls_back(self):
        from src.proj import PATH

        with (
            TemporaryDirectory() as tmp,
            patch.object(PATH, "result", Path(tmp)),
            patch.object(PATH, "lc_share", Path(tmp) / "shared"),
        ):
            fixture(Path(tmp), status="failed")
            with self.assertRaisesRegex(ValueError, "No completed repaired dataset"):
                resolve_schedule(
                    "repaired_missing", "factorseq_residual", "test", "snapshot"
                )

    def test_expansion_purge_and_label_holes(self):
        dates = np.arange(20000000, 20002500)
        ends = dates + 11
        ready = np.ones(len(dates), bool)
        folds = expanded_folds(dates, ends, ready, start=int(dates[1000]))
        self.assertLess(folds[0]["training_days_before_purge"], 1260)
        self.assertTrue(folds[-1]["full_window"])
        for fold in folds:
            self.assertLess(max(fold["train"]) + 11, min(fold["valid"]))
            self.assertLess(max(fold["valid"]) + 11, min(fold["test"]))
        ready[1500] = False
        with self.assertRaisesRegex(ValueError, "Missing daily labels"):
            expanded_folds(dates, ends, ready, start=int(dates[1000]))

    def test_worklist_order_and_common_source(self):
        root = Path(__file__).resolve().parents[1]
        work = yaml.safe_load((root / "configs/schedule/worklist.yaml").read_text())
        names = [row["name"] for row in work["fit"]]
        expected = [f"factorseq_v3_{m}" for m in ("snapshot", "temporal", "contextual")]
        self.assertIn("resnet_gru_simp_15m", names)
        self.assertEqual([n for n in names if n.startswith("factorseq_")], expected)
        sources = []
        for name in expected:
            config = yaml.safe_load(
                (root / f"configs/schedule/current/{name}.yaml").read_text()
            )
            sources.append(config["factor_sequence_source"])
        self.assertEqual(len(set(sources)), 1)
        self.assertTrue(sources[0].startswith("repaired_"))


if __name__ == "__main__":
    unittest.main()
