"""Microcap repair rejects silent-zero inputs and preserves outputs on failure."""

from __future__ import annotations
from src.data.maintenance import transaction

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.maintenance import repair_microcap as repair
from src.data.util.prices import adjusted as adjprice

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

    def test_missing_returns_print_both_endpoints_and_save_evidence(self):
        weights = pd.DataFrame({"secid": [1, 2, 3], "weight": [1/3]*3})
        returns = pd.DataFrame({"secid": [1, 2], "ret": [.01, np.nan]}).set_index("secid")
        day0 = pd.DataFrame({"secid": [1, 2, 3], "close": [10., 20., 30.],
                             "adjfactor": [1., 2., np.nan], "status": [1, 1, 1], "volume": [1., 1., 1.]})
        day1 = pd.DataFrame({"secid": [1, 2], "close": [10.1, 20.],
                             "adjfactor": [1., 2.], "status": [1, 0], "volume": [1., 0.]})
        with tempfile.TemporaryDirectory() as name:
            with (patch.object(repair.DB, "load", side_effect=[day0, pd.DataFrame(), day1,
                      pd.DataFrame({"secid": [2], "close": [40.]})]) as load,
                  patch.object(repair.Logger, "stdout") as log):
                with self.assertRaisesRegex(ValueError, "missing/nonfinite holdings returns"):
                    repair.checked_return(weights, returns, 20050104, previous=20041231, reports=Path(name))
            evidence = pd.read_csv(Path(name) / 'missing_returns_20041231_20050104.csv')
            self.assertEqual(len(evidence), 4)
            self.assertEqual(set(evidence.secid), {2, 3})
            self.assertEqual(set(evidence.date), {20041231, 20050104})
            start = evidence.query("secid == 3 and endpoint == 'start'").iloc[0]
            self.assertEqual(start.raw_adjusted_close, 30.)
            self.assertTrue(start.adjprice_file_empty)
            self.assertFalse(start.return_row_present)
            end = evidence.query("secid == 2 and endpoint == 'end'").iloc[0]
            self.assertEqual(end.adjprice_close, 40.)
            self.assertTrue(end.return_row_present)
            self.assertIn('suspension flag', end.suspension_evidence)
            absent = evidence.query("secid == 3 and endpoint == 'end'").iloc[0]
            self.assertEqual(absent.day_row_count, 0)
            self.assertIn('unknown', absent.suspension_evidence)
            self.assertEqual(load.call_count, 4)
            printed = '\n'.join(str(call.args[0]) for call in log.call_args_list)
            self.assertIn('"secid": 2', printed)
            self.assertIn('"secid": 3', printed)
            self.assertIn('"date": 20041231', printed)
            self.assertIn('"date": 20050104', printed)

    def test_diagnostic_read_errors_do_not_hide_missing_returns(self):
        weights = pd.DataFrame({"secid": [1], "weight": [1.]})
        returns = pd.DataFrame(columns=["secid", "ret"]).astype({"ret": float})
        with tempfile.TemporaryDirectory() as name:
            with (patch.object(repair.DB, "load", side_effect=OSError('unreadable input')),
                  patch.object(repair.Logger, "stdout")):
                with self.assertRaisesRegex(ValueError, "missing/nonfinite holdings returns"):
                    repair.checked_return(weights, returns, 20050104, previous=20041231, reports=Path(name))
            evidence = pd.read_csv(Path(name) / 'missing_returns_20041231_20050104.csv')
            self.assertTrue(evidence.day_read_error.str.contains('unreadable input').all())
            self.assertTrue(evidence.adjprice_read_error.str.contains('unreadable input').all())

    def test_valid_returns_do_not_read_diagnostic_inputs(self):
        weights = pd.DataFrame({"secid": [1], "weight": [1.]})
        returns = pd.DataFrame({"secid": [1], "ret": [.02]})
        with patch.object(repair.DB, "load") as load:
            self.assertEqual(repair.checked_return(weights, returns, 20050104, previous=20041231), 2.)
        load.assert_not_called()

    def test_stale_adjprice_rejected(self):
        day = pd.DataFrame({"secid": [1], "close": [11.0], "adjfactor": [2.0]})
        adj = pd.DataFrame({"secid": [1], "close": [20.0]})
        with (patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=["secid", "list_dt", "delist_dt"])),
              patch.object(repair.DB, "load", side_effect=[day, adj])):
            with self.assertRaisesRegex(ValueError, "stale adjprice"):
                repair.check_prices(20261118)
        with (patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=["secid", "list_dt", "delist_dt"])),
              patch.object(repair.DB, "load", side_effect=[day, adj.assign(close=22.0)])):
            repair.check_prices(20261118)

    def test_price_check_matches_adjusted_price_listed_universe(self):
        day = pd.DataFrame({"secid": [1, 2], "close": [11.0, 5.0], "adjfactor": [2.0, 1.0]})
        adj = pd.DataFrame({"secid": [1], "close": [22.0]})
        desc = pd.DataFrame({"secid": [2], "list_dt": [20100101], "delist_dt": [99991231]})
        with (patch.object(adjprice, 'listing_description', return_value=desc),
              patch.object(repair.DB, "load", side_effect=[day, adj])):
            repair.check_prices(20091231)
        # A missing *listed* security must still fail, with actionable evidence.
        with (patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=["secid", "list_dt", "delist_dt"])),
              patch.object(repair.DB, "load", side_effect=[day, adj])):
            with self.assertRaisesRegex(ValueError, r"1/2 listed quotes differ.*1 missing.*secid=2"):
                repair.check_prices(20091231)

    def test_missing_adjfactor_matches_production_with_and_without_adjprice(self):
        day = pd.DataFrame({"secid": [1, 2], "close": [11.0, 5.0], "adjfactor": [np.nan, 2.0]})
        valid = pd.DataFrame({"secid": [1, 2], "close": [11.0, 10.0]})
        for adj in (valid, pd.DataFrame()):
            with (self.subTest(adjprice_present=not adj.empty),
                  patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=["secid", "list_dt", "delist_dt"])),
                  patch.object(repair.DB, "load", side_effect=[day, adj])):
                repair.check_prices(20041231)
        with (patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=["secid", "list_dt", "delist_dt"])),
              patch.object(repair.DB, "load", side_effect=[day, valid.assign(close=[12., 10.])])):
            with self.assertRaisesRegex(ValueError, "stale adjprice.*secid=1"):
                repair.check_prices(20041231)

    def test_invalid_raw_fields_still_fail_with_security_evidence(self):
        for close, factor in ((np.nan, 1.), (0., 1.), (-1., 1.), (np.inf, 1.),
                              (10., 0.), (10., -1.), (10., np.inf), (-10., -1.)):
            day = pd.DataFrame({"secid": [123], "close": [close], "adjfactor": [factor]})
            with (self.subTest(close=close, factor=factor),
                  patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=["secid", "list_dt", "delist_dt"])),
                  patch.object(repair.DB, "load", return_value=day)):
                with self.assertRaisesRegex(ValueError, r"20041231:.*1/1 listed quotes invalid.*secid=123, close=.*adjfactor="):
                    repair.check_prices(20041231)

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
            original = transaction.os.replace

            def fail_second(source, destination):
                if destination == new_target:
                    raise OSError("injected failure")
                original(source, destination)

            with patch.object(transaction.os, "replace", side_effect=fail_second):
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
