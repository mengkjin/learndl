from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.res.rl_experimental.bundle import analyze_bundle, create_bundle, send_bundle
from src.res.rl_experimental.data import make_synthetic_panel
from src.res.rl_experimental.env import EpisodeRange, PortfolioEnv
from src.res.rl_experimental.experiment import run_experiment
from src.res.rl_experimental.portfolio import PortfolioConstraints
from src.res.rl_experimental.reward import RewardConfig, RewardContext, RewardEvaluator


class RewardTest(unittest.TestCase):
    def _context(self) -> RewardContext:
        return RewardContext(
            decision_date=20250101,
            return_end_date=20250102,
            observation={"account": np.array([1.0, 0.0], dtype=np.float32)},
            current_weights=np.array([0.4, 0.4]),
            target_weights=np.array([0.5, 0.5]),
            end_weights=np.array([0.6, 0.3]),
            end_cash=0.1,
            nav_multiplier=0.98,
            net_return=-0.02,
            turnover=0.5,
            fee_fraction=0.001,
            execution={"rejected_buys": 0},
        )

    def test_default_and_parameterized_reward_are_hand_checkable(self) -> None:
        context = self._context()
        legacy = RewardEvaluator(RewardConfig(), 100.0)(context)
        self.assertAlmostEqual(legacy.reward, 100 * np.log(0.98))
        configured = RewardEvaluator(
            RewardConfig(turnover_penalty=0.01, downside_penalty=2.0, concentration_penalty=0.1), 100.0,
        )(context)
        expected = 100 * (np.log(0.98) - 0.01 * 0.5 - 2.0 * 0.02**2 - 0.1 * (0.6**2 + 0.3**2))
        self.assertAlmostEqual(configured.reward, expected)
        self.assertAlmostEqual(configured.reward, sum(configured.components.values()))

    def test_environment_records_reward_components_without_double_charging_fee(self) -> None:
        panel = make_synthetic_panel(20, 12, 3, seed=4)
        env = PortfolioEnv(
            panel, EpisodeRange(0, 10), PortfolioConstraints(6, 3, 0.4, fee_rate=0.001),
            reward_config=RewardConfig(turnover_penalty=0.01),
        )
        env.reset(seed=1)
        _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        expected = 100 * (np.log(info["period_return"] + 1.0) - 0.01 * info["turnover"])
        self.assertAlmostEqual(reward, expected)
        self.assertIn("reward_component/log_growth", info)
        self.assertIn("reward_component/turnover_penalty", info)
        self.assertGreaterEqual(info["fee_estimate"], 0.0)

    def test_custom_reward_metadata_and_nonfinite_result(self) -> None:
        config = RewardConfig(
            custom_function="src.res.rl_experimental.reward:example_drawdown_aware_reward",
            custom_params={"downside_penalty": 3.0, "turnover_penalty": 0.002},
        )
        evaluator = RewardEvaluator(config, 10.0)
        result = evaluator(self._context())
        self.assertTrue(np.isfinite(result.reward))
        self.assertEqual(evaluator.metadata["entrypoint"], config.custom_function)
        self.assertIsNotNone(evaluator.metadata["source_sha256"])


class BundleTest(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        run = root / "run-1"
        run.mkdir()
        (run / "status.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
        (run / "metrics.json").write_text("{}", encoding="utf-8")
        (run / "report.html").write_text("<h1>ok</h1>", encoding="utf-8")
        (run / "best_model.zip").write_bytes(b"model" * 80)
        (run / "panel.npz").write_bytes(b"must not ship")
        return run

    def test_split_verify_extract_and_exclude_panel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = create_bundle(self._run(root), root / "bundles", max_part_bytes=120)
            definition = json.loads(metadata.read_text())
            self.assertGreater(len(definition["parts"]), 1)
            extracted = analyze_bundle(metadata, root / "received")
            self.assertTrue((extracted / "report.html").is_file())
            self.assertFalse((extracted / "panel.npz").exists())
            self.assertTrue((extracted / "local_analysis.json").is_file())

    def test_corrupt_part_and_unsafe_archive_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = create_bundle(self._run(root), root / "bundles", max_part_bytes=10_000)
            definition = json.loads(metadata.read_text())
            part = metadata.parent / definition["parts"][0]["name"]
            part.write_bytes(part.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "size/hash"):
                analyze_bundle(metadata, root / "received")

            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr("../escape", "bad")
            import hashlib

            unsafe_meta = root / "unsafe.parts.json"
            unsafe_meta.write_text(json.dumps({
                "run_id": "unsafe", "archive_name": "unsafe.zip",
                "archive_sha256": hashlib.sha256(unsafe.read_bytes()).hexdigest(),
                "parts": [{"name": "unsafe.zip", "size": unsafe.stat().st_size,
                           "sha256": hashlib.sha256(unsafe.read_bytes()).hexdigest()}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe bundle member"):
                analyze_bundle(unsafe_meta, root / "unsafe-output")

    def test_delivery_retries_only_failed_parts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = create_bundle(self._run(root), root / "bundles", max_part_bytes=120)
            calls: list[str] = []

            def first(_title, _body, _recipient, attachments):
                calls.append(attachments[0].name)
                return len(calls) != 1

            state = send_bundle(metadata, "user@example.com", transport=first)
            failed = [name for name, item in state["parts"].items() if not item["sent"]]
            self.assertEqual(len(failed), 1)
            retried: list[str] = []

            def second(_title, _body, _recipient, attachments):
                retried.append(attachments[0].name)
                return True

            final = send_bundle(metadata, "user@example.com", only_failed=True, transport=second)
            self.assertEqual(retried, failed)
            self.assertTrue(all(item["sent"] for item in final["parts"].values()))

    def test_failed_experiment_still_produces_diagnostic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({
                "alpha": "pred@missing", "start": 20250102, "end": 20250228,
                "snapshot_dir": str(root / "snapshot"), "output_root": str(root / "output"),
                "training": {"device": "cpu"},
            }), encoding="utf-8")
            ready = []
            with patch(
                "src.res.rl_experimental.experiment._snapshot", side_effect=FileNotFoundError("missing alpha"),
            ), self.assertRaisesRegex(RuntimeError, "diagnostic bundle"):
                run_experiment(config, no_email=True, bundle_ready=ready.append)
            metadata = next((root / "output/bundles").glob("*.parts.json"))
            self.assertEqual(ready, [metadata.resolve()])
            definition = json.loads(metadata.read_text())
            part = metadata.parent / definition["parts"][0]["name"]
            with zipfile.ZipFile(part) as archive:
                status = json.loads(archive.read("status.json"))
                self.assertEqual(status["status"], "failed")
                self.assertIn("failure_report.html", archive.namelist())


if __name__ == "__main__":
    unittest.main()
