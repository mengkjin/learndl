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
from src.data.preprocess.factor_sequence.repaired_bundle import REPAIR_VERSION
from src.res.model.experiment.factor_sequence.repaired_schedule import expanded_folds
from src.res.model.experiment.factor_sequence.worklist import resolve_schedule
from src.res.algo.nn.api import get_nn_module
from src.res.model.util.data.module_data import ModuleData
from src.res.model.util.data.data_module import DataModule

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
    def test_v3_and_v4_schedules_pin_same_data_and_use_compact_inputs(self):
        from src.proj import PATH, CALENDAR

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            dataset, dates = fixture(root)
            stack.enter_context(patch.object(PATH, "rslt_custom", root))
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
                    "src.res.model.experiment.factor_sequence.repaired_schedule._label_state",
                    side_effect=lambda _, ds: (np.ones(len(ds), bool), "stable_labels"),
                )
            )
            frozen_folds = None
            modes = {
                "snapshot": ("factorseq_v3_snapshot", [132]),
                "temporal": ("factorseq_v3_temporal", [132]),
                "contextual": ("factorseq_v3_contextual", [132, 132, 30]),
                "contextual_v4": ("factorseq_v4_contextual", [132, 132]),
                "contextual_v4_market": (
                    "factorseq_v4_contextual_market",
                    [132, 132, 30],
                ),
                "contextual_v4_events": (
                    "factorseq_v4_contextual_events",
                    [132, 132, 30],
                ),
                **{
                    variant: (f"factorseq_v4_{variant}", [132, 132])
                    for variant in ("shared", "private", "graph")
                },
            }
            for mode, (name, expected_widths) in modes.items():
                v4_sequence = mode in ("shared", "private", "graph")
                module = (
                    "factorseq_contextual_v4" if v4_sequence else "factorseq_residual"
                )
                config = resolve_schedule(
                    "repaired_20261117", module, "test_" + mode, mode
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
                from src.res.model.util.config.config import (
                    AlgoConfig,
                    ModelConfig,
                    ScheduleConfig,
                )

                if mode.startswith("contextual_v4"):
                    archive = PATH.sched_archive / f"{name}.yaml"
                    self.assertIsNone(ScheduleConfig.find_path(name=name))
                    # An archived stub still resolves to the same legacy weights/config.
                    with patch.object(
                        ScheduleConfig, "find_path", return_value=archive
                    ):
                        effective = ScheduleConfig(schedule_name=name)
                else:
                    effective = ScheduleConfig(schedule_name=name)
                self.assertEqual(
                    effective["input.factor_sequence.dataset"], str(dataset.resolve())
                )
                variant_key = "variant" if v4_sequence else "mode"
                self.assertEqual(effective[f"algo.{module}.{variant_key}"], [mode])
                self.assertEqual(effective["model.name"], name)
                if frozen_folds is None:
                    frozen_folds = effective["train.factor_sequence.folds"]
                self.assertEqual(effective["train.factor_sequence.folds"], frozen_folds)
                self.assertEqual(config["env.random_seed"], 17)
                keys = config["input.data.types"].split("+")
                self.assertEqual(len(keys), len(expected_widths))
                lens = config["input.sequence.lens"]
                data = ModuleData.initialize(
                    keys,
                    y_labels=effective["model.labels"],
                    factor_sequence_dataset=str(dataset),
                    factor_sequence_view=effective["input.factor_sequence.view"],
                    vb_level="never",
                )
                self.assertEqual(data.data_type_list, keys)
                data.blocks.update(
                    {
                        k: data.load_one(
                            k, dates=dates[1600 - lens[k] : 1600], secid=[1, 2]
                        )
                        for k in keys
                    }
                )
                inputs = [data.blocks[k].values for k in keys]
                widths = [x.shape[-1] for x in inputs]
                self.assertEqual(widths, expected_widths)
                algo = AlgoConfig(
                    None,
                    start_with_none=True,
                    module=module,
                    schedule_config=effective,
                    vb_level="never",
                ).expand()
                self.assertEqual(algo.n_model, 1)
                if mode.startswith("contextual_v4"):
                    self.assertEqual(
                        algo.params[0]["gate_half_lives"], [5.0, 20.0, 60.0]
                    )
                    self.assertEqual(algo.params[0]["gate_penalty"], 0.01)
                    expected_lens = {"factorseq": 60, "factorperf": 120}
                    if mode != "contextual_v4":
                        expected_lens["factormarket"] = 1
                        self.assertEqual(algo.params[0]["market_rank"], 2)
                    if mode == "contextual_v4_events":
                        self.assertEqual(algo.params[0]["event_rank"], 1)
                        self.assertEqual(algo.params[0]["event_decay_days"], 20.0)
                    self.assertEqual(keys, list(expected_lens))
                    self.assertEqual(lens, expected_lens)
                if v4_sequence:
                    self.assertEqual(keys, ["factorseq", "factorperf"])
                    self.assertEqual(lens, {"factorseq": 60, "factorperf": 60})
                    self.assertEqual(tuple(inputs[1].shape), (1, 60, 1, 132))
                    model_config = ModelConfig(
                        schedule_name=name,
                        stage=0,
                        resume=0,
                        selection=0,
                        vb_level="never",
                    )
                    self.assertEqual(model_config.base_path.model_clean_name, name)
                    data_module = DataModule(model_config)
                    data_module.datas = data
                    with patch.object(
                        data_module, "no_need_to_load_data", return_value=True
                    ):
                        self.assertIs(data_module.load_data(), data_module)
                    data_module.setup_new_param("test", {}, 20161230)
                    streams = data_module.batch_data_x(
                        {k: x for k, x in zip(keys, inputs)},
                        torch.arange(2),
                        torch.full((2,), 59),
                    )
                    self.assertEqual(tuple(streams[1].shape), (1, 60, 132))
                    inputs = streams
                # Use the same registry as the ordinary trainer.
                network = get_nn_module(module)(widths, **algo.params[0])
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
                "src.res.model.experiment.factor_sequence.repaired_schedule._label_state",
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
            patch.object(PATH, "rslt_custom", Path(tmp)),
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

    def test_current_schedule_sources_and_legacy_suite_membership(self):
        from src.res.model.experiment.factor_sequence.schedule_training import variants

        root = Path(__file__).resolve().parents[1]
        expected = [f"factorseq_v3_{m}" for m in ("snapshot", "temporal", "contextual")]
        expected += [
            f"factorseq_v4_{variant}" for variant in ("shared", "private", "graph")
        ]
        self.assertEqual(
            set(dict(variants())),
            {
                "factorseq_mlp",
                "factorseq_gru",
                "factorseq_m1",
                "factorseq_m2",
                "factorseq_mask",
                "factorseq_gate",
            },
        )
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
