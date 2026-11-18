"""Microcap repair rejects silent-zero inputs and preserves outputs on failure."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.preprocess import repair_microcap as repair

__all__ = ["MicrocapRepairTest"]


class MicrocapRepairTest(unittest.TestCase):
    def test_end_to_end_market_rebuild_and_validate_only(self):
        for validate_only in (True, False):
            with (
                self.subTest(validate_only=validate_only),
                tempfile.TemporaryDirectory() as name,
            ):
                root = Path(name)
                target = root / "microcap_400.feather"
                old = pd.DataFrame(
                    {"trade_date": [20261117, 20261118], "pct_chg": [0.0, 0.0]}
                )
                old.to_feather(target)
                weights = pd.DataFrame({"secid": [1], "weight": [1.0]})
                returns = pd.DataFrame({"secid": [1], "ret": [0.02]})
                index = SimpleNamespace(
                    rebalance_portfolio=lambda d: weights, index_return=lambda d: 0.02
                )
                paths = SimpleNamespace(
                    result=root / "results", datacache=root / "cache"
                )

                def dump_path(key, frame):
                    return root / (frame + ".pt")

                with (
                    patch.object(repair, "PATH", paths),
                    patch.object(repair.DB, "path", return_value=target),
                    patch.object(repair.DB, "load", return_value=old),
                    patch.object(repair.CALENDAR, "updated", return_value=20261118),
                    patch.object(
                        repair.CALENDAR,
                        "range",
                        return_value=np.array([20261117, 20261118]),
                    ),
                    patch.object(
                        repair.CALENDAR,
                        "td",
                        return_value=SimpleNamespace(
                            td=20261116, as_int=lambda: 20261116
                        ),
                    ),
                    patch.object(repair, "check_prices"),
                    patch.object(repair, "MicroCap_400", return_value=index) as factory,
                    patch.object(
                        repair.DATAVENDOR, "get_quote_ret", return_value=returns
                    ),
                    patch.object(repair.DATAVENDOR.TRADE, "clear_all"),
                    patch.object(
                        repair.DataBlock, "path_preprocess", side_effect=dump_path
                    ),
                    patch.object(
                        repair.PrePro_market, "start_date", return_value=20261117
                    ),
                    patch.object(
                        repair.PrePro_market,
                        "_load_risk_frame",
                        return_value=pd.DataFrame(
                            {
                                key: [0.0, 0.0]
                                for key in repair.PrePro_market.RISK_FEATURES
                            },
                            index=[20261117, 20261118],
                        ),
                    ),
                    patch.object(
                        repair.PrePro_market,
                        "_load_index_logret",
                        return_value=pd.Series([0.0, 0.0], index=[20261117, 20261118]),
                    ),
                    patch("src.automation.artifact_exchange.research.register"),
                ):
                    factory.START_DATE = 20100101
                    report = repair.rebuild_microcap(20261117, 20261118, validate_only)
                self.assertEqual(
                    report["status"], "validated" if validate_only else "complete"
                )
                self.assertEqual(
                    pd.read_feather(target).pct_chg.tolist(),
                    [0.0, 0.0] if validate_only else [2.0, 2.0],
                )
                self.assertEqual((root / "fit.pt").exists(), not validate_only)
                self.assertEqual((root / "predict.pt").exists(), not validate_only)

    def test_market_block_roundtrip_all_supported_formats(self):
        frame = pd.DataFrame(
            {
                "secid": [0, 0],
                "date": [20261117, 20261118],
                "logret_microcap_400": [0.01, -0.02],
            }
        )
        block = repair.DataBlock.from_pandas(frame)
        with tempfile.TemporaryDirectory() as name:
            for suffix in (".mmap", ".pt", ".feather"):
                with self.subTest(suffix=suffix):
                    repair.save_block(block, Path(name) / ("market" + suffix))

    def test_checked_return_requires_complete_holdings(self):
        weights = pd.DataFrame({"secid": [1, 2], "weight": [0.5, 0.5]})
        returns = pd.DataFrame({"secid": [1, 2], "ret": [0.1, -0.02]})
        self.assertAlmostEqual(repair.checked_return(weights, returns, 20261118), 4.0)
        for invalid in (
            returns.iloc[:1],
            returns.assign(ret=np.nan),
            pd.concat([returns, returns]),
        ):
            with self.assertRaises(ValueError):
                repair.checked_return(weights, invalid, 20261118)
        with self.assertRaises(ValueError):
            repair.checked_return(weights.iloc[:0], returns, 20261118)
        self.assertEqual(
            repair.checked_return(weights, returns.assign(ret=0.0), 20261118), 0.0
        )

    def test_stale_adjprice_rejected(self):
        day = pd.DataFrame({"secid": [1], "close": [11.0], "adjfactor": [2.0]})
        adj = pd.DataFrame({"secid": [1], "close": [20.0]})
        with (patch.object(repair.INFO, "get_secid", return_value=np.array([1])),
              patch.object(repair.DB, "load", side_effect=[day, adj])):
            with self.assertRaisesRegex(ValueError, "stale adjprice"):
                repair.check_prices(20261118)
        with (patch.object(repair.INFO, "get_secid", return_value=np.array([1])),
              patch.object(repair.DB, "load", side_effect=[day, adj.assign(close=22.0)])):
            repair.check_prices(20261118)

    def test_price_check_matches_adjusted_price_listed_universe(self):
        day = pd.DataFrame({"secid": [1, 2], "close": [11.0, 5.0], "adjfactor": [2.0, 1.0]})
        adj = pd.DataFrame({"secid": [1], "close": [22.0]})
        with (patch.object(repair.INFO, "get_secid", return_value=np.array([1])),
              patch.object(repair.DB, "load", side_effect=[day, adj])):
            repair.check_prices(20091231)
        # A missing *listed* security must still fail, with actionable evidence.
        with (patch.object(repair.INFO, "get_secid", return_value=np.array([1, 2])),
              patch.object(repair.DB, "load", side_effect=[day, adj])):
            with self.assertRaisesRegex(ValueError, r"1/2 listed quotes differ.*1 missing.*secid=2"):
                repair.check_prices(20091231)

    def test_numpy_dates_do_not_mask_original_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with (patch.object(repair, "PATH", SimpleNamespace(result=root)),
                  patch.object(repair.CALENDAR, "updated", return_value=np.int64(20261118)),
                  patch.object(repair.DB, "path", side_effect=ValueError("original price failure")),
                  patch("src.automation.artifact_exchange.research.register") as register):
                with self.assertRaisesRegex(ValueError, "original price failure"):
                    repair.rebuild_microcap(np.int64(20100101))
            summaries = list(root.rglob("summary.json"))
            self.assertEqual(len(summaries), 1)
            report = json.loads(summaries[0].read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["start"], 20100101)
            self.assertEqual(report["end"], 20261118)
            self.assertEqual(report["error"], "ValueError: original price failure")
            register.assert_called_once()
            self.assertFalse(list(root.rglob("transaction.json")))

    def test_only_market_module_caches_selected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            records = {
                "a": dict(type="module_data", data_type_list=["day", "market"]),
                "b": dict(type="module_data", data_type_list=["factormarket"]),
                "c": dict(type="other", data_type_list=["market"]),
            }
            (root / "cache_metadata.json").write_text(json.dumps(records))
            for key in records:
                (root / key).mkdir()
                (root / key / "data.pt").write_bytes(b"cached")
            self.assertEqual(
                repair.market_cache_paths(root), [(root / "a" / "data.pt").resolve()]
            )

    def test_install_replaces_file_directory_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            target, stage = root / "index", root / "candidate"
            target.write_text("old")
            stage.write_text("new")
            cache = root / "cache.pt"
            cache.write_text("stale")
            block, block_stage = root / "block", root / "block-stage"
            block.mkdir()
            block_stage.mkdir()
            (block / "values").write_text("old")
            (block_stage / "values").write_text("new")
            baseline = {p: repair.digest(p) for p in (target, block, cache)}
            repair.install(
                [(target, stage), (block, block_stage), (cache, None)],
                baseline,
                root / "run",
            )
            self.assertEqual(target.read_text(), "new")
            self.assertEqual((block / "values").read_text(), "new")
            self.assertFalse(cache.exists())
            self.assertEqual((root / "run/backups/0").read_text(), "old")

    def test_install_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            target, stage, new_target = root / "index", root / "candidate", root / "new"
            target.write_text("old")
            stage.write_text("new")
            baseline = {p: repair.digest(p) for p in (target, new_target)}
            original = repair.os.replace

            def fail_second(source, destination):
                if destination == new_target:
                    raise OSError("injected failure")
                original(source, destination)

            with patch.object(repair.os, "replace", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    repair.install(
                        [(target, stage), (new_target, stage)], baseline, root / "run"
                    )
            self.assertEqual(target.read_text(), "old")
            self.assertFalse(new_target.exists())

    def test_concurrent_output_change_prevents_install(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            target = root / "target"
            target.write_text("initial")
            baseline = {target: repair.digest(target)}
            target.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "changed during repair"):
                repair.install([(target, None)], baseline, root / "run")
            self.assertEqual(target.read_text(), "changed")
