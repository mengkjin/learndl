"""Offline repair contracts: ties, missingness, regression and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from src.func.factor_groups import factor_groups
from src.data.preprocess.factor_sequence.dividend import dividend_values
from src.data.maintenance.factor_sequence.repair import Checkpoint, dependency_order
from src.data.preprocess.factor_sequence.numeric import (
    stock_block,
    group_metrics,
    market_block,
)
from src.data.preprocess.factor_sequence.schema import BuildConfig
from src.res.factor.defs.market.level0.market.event import high_level_switch
from src.res.factor.defs.stock.level0.behavior.behavior_ff3 import FF3_Model
from src.res.factor.calculator.factor_calc import MarketFactorCalculator
from src.res.factor.util.classes.stock_factor import StockFactor

__all__ = ["RepairTests"]


class RepairTests(unittest.TestCase):
    def test_streaming_build_chunk_and_future_invariance(self):
        from src.res.model.experiment.factor_sequence.sources import build_repaired_project
        from src.data.preprocess.factor_sequence.context_audit import RISK_NAMES
        from src.data.preprocess.factor_sequence.schema import EVENTS
        from src.proj import DB, CALENDAR
        from src.data import DATAVENDOR
        from src.res.factor.calculator import StockFactorHierarchy
        from contextlib import ExitStack

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            source = root / "source"
            source.write_text("fixed fixture source")
            ids = np.arange(60)
            calendar = np.arange(250)
            risk = pd.DataFrame(
                {
                    "date": calendar,
                    **{name: np.sin(calendar / 10) + 2 for name in RISK_NAMES},
                }
            )
            quote = pd.DataFrame({"trade_date": calendar, "pct_chg": np.sin(calendar)})
            event = pd.DataFrame(
                {
                    "date": calendar,
                    **{name: (calendar % 20 == 0).astype(float) for name in EVENTS},
                }
            )
            provenance = pd.DataFrame(
                {
                    "secid": ids,
                    "dtop__source_observed": (ids % 2 == 0).astype(float),
                    "dtop__source_filled": (ids % 2 != 0).astype(float),
                }
            )

            def load(src, key, date=None, **kw):
                if src == "market_daily":
                    return risk.copy()
                if src == "index_daily_ts":
                    return quote.copy()
                return provenance.copy()

            def factor(name):
                return SimpleNamespace(
                    db_src="stock_factor", db_key=name, Load=lambda _: event.copy()
                )

            def read(calc, name, date, universe, db, first):
                value = ids.astype(float) + np.sin(date + ids)
                if name == "dtop":
                    value = np.where(ids % 2 == 0, 0.01, 0.0)
                if name == "a":
                    value[0] = np.nan
                return pd.Series(value, index=ids), dict(
                    date=date, status="available", factor=name
                )

            def prices(start, end, *args, **kw):
                return pd.DataFrame(
                    [np.ones(60), 1 + ids / 10000], index=[start, end], columns=ids
                )

            stack.enter_context(
                patch(
                    "src.res.model.experiment.factor_sequence.sources.BuildConfig",
                    lambda: BuildConfig(factors=("a", "dtop")),
                )
            )
            stack.enter_context(
                patch(
                    "src.res.model.experiment.factor_sequence.audit_runner.read_factor",
                    side_effect=read,
                )
            )
            stack.enter_context(
                patch.object(StockFactorHierarchy, "get_factor", side_effect=factor)
            )
            stack.enter_context(patch.object(DB, "load", side_effect=load))
            stack.enter_context(patch.object(DB, "path", return_value=source))
            stack.enter_context(
                patch.object(
                    CALENDAR, "range", side_effect=lambda a, b, *_: np.arange(a, b + 1)
                )
            )
            stack.enter_context(
                patch.object(CALENDAR, "td", side_effect=lambda a, b: a + b)
            )
            stack.enter_context(
                patch.object(
                    DATAVENDOR.INFO,
                    "get_list_dt",
                    return_value=pd.DataFrame(
                        {"list_dt": 0, "delist_dt": 9999}, index=ids
                    ),
                )
            )
            stack.enter_context(
                patch.object(DATAVENDOR.TRADE, "get_specific_data", side_effect=prices)
            )
            stack.enter_context(patch.object(DATAVENDOR, "data_storage_control"))
            a = build_repaired_project(root / "a", 180, 190, batch_size=4)
            b = build_repaired_project(root / "b", 180, 190, batch_size=7)
            c = build_repaired_project(root / "c", 180, 195, batch_size=4)
            for key in ("factorseq", "factorperf", "factormarket"):
                x = np.load(a / f"{key}.npy")
                np.testing.assert_allclose(x, np.load(b / f"{key}.npy"))
                np.testing.assert_allclose(x, np.load(c / f"{key}.npy")[:, :11])
            stock = np.load(a / "factorseq.npy")
            self.assertTrue((stock[0, :, :, 2] == 0).all())  # a remains unknown
            self.assertTrue(
                (stock[1, :, :, 3] == 1).all()
            )  # dtop economic zero is valid

    def test_dry_run_and_failed_delivery_registration(self):
        from src.data.maintenance.factor_sequence.repair import run_repair
        from src.proj import PATH, CALENDAR, DB
        from contextlib import ExitStack

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(PATH, "rslt_custom", Path(tmp)))
            stack.enter_context(
                patch.object(CALENDAR, "updated", return_value=20261113)
            )
            stack.enter_context(
                patch(
                    "src.data.maintenance.factor_sequence.repair._inventory",
                    return_value="raw",
                )
            )
            stack.enter_context(
                patch(
                    "src.data.maintenance.factor_sequence.repair._factor_inventory",
                    return_value="factors",
                )
            )
            save = stack.enter_context(patch.object(DB, "save"))
            build = stack.enter_context(
                patch("src.data.maintenance.factor_sequence.repair.build_repaired_project")
            )
            register = stack.enter_context(
                patch("src.automation.artifact_exchange.research.register")
            )
            folder = run_repair("backfill", run_id="dry", dry_run=True)
            self.assertEqual(
                json.loads((folder / "reports" / "summary.json").read_text())["status"],
                "dry_run",
            )
            save.assert_not_called()
            build.assert_not_called()
            self.assertEqual(
                Path(register.call_args.args[0].source), folder / "reports"
            )
            with patch(
                "src.data.maintenance.factor_sequence.repair.Checkpoint",
                side_effect=RuntimeError("source revised"),
            ):
                with self.assertRaisesRegex(RuntimeError, "source revised"):
                    run_repair("backfill", run_id="failure")
            summary = json.loads(
                (
                    Path(tmp)
                    / "factor_sequence_fixes"
                    / "failure"
                    / "reports"
                    / "summary.json"
                ).read_text()
            )
            self.assertEqual(summary["status"], "failed")
            save.assert_not_called()
            build.assert_not_called()

    def test_cross_run_verified_reuse(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "factor"
            identity = {"code": "code", "raw_sources": "source", "version": "v"}
            first = Checkpoint(root / "one", identity, root / "reuse.sqlite")
            first.step("factor:dtop:20110104", [out], lambda: out.write_text("valid"))
            first.db.close()
            assert first.reuse is not None
            first.reuse.close()
            second = Checkpoint(root / "two", identity, root / "reuse.sqlite")
            self.assertFalse(
                second.step(
                    "factor:dtop:20110104",
                    [out],
                    lambda: self.fail("unexpected recompute"),
                )
            )
            second.db.close()
            assert second.reuse is not None
            second.reuse.close()
            out.write_text("changed")
            third = Checkpoint(root / "three", identity, root / "reuse.sqlite")
            self.assertTrue(
                third.step(
                    "factor:dtop:20110104", [out], lambda: out.write_text("valid")
                )
            )
            third.db.close()
            assert third.reuse is not None
            third.reuse.close()

    def test_dividend_zero_provenance(self):
        frame = dividend_values(
            pd.DataFrame({"secid": [1, 2], "dv_ttm": [2.0, np.nan]}), [1, 2, 3]
        )
        np.testing.assert_allclose(frame.dtop, [0.02, 0, 0])
        self.assertEqual(frame.dtop__source_observed.tolist(), [1, 0, 0])
        self.assertEqual(frame.dtop__source_filled.tolist(), [0, 1, 1])
        with self.assertRaises(ValueError):
            dividend_values(pd.DataFrame(), [1])
        with self.assertRaises(ValueError):
            dividend_values(pd.DataFrame({"secid": [1, 1], "dv_ttm": [1, 2]}), [1])

    def test_group_parity_and_minimum(self):
        x = np.r_[np.zeros(70), np.ones(29), 100.0]
        r = np.arange(100) / 1000
        result = group_metrics(x, r, 50)
        frame = pd.DataFrame({"date": 1, "f": x, "ret": r})
        groups = StockFactor._eval_group_perf(frame, ["f"])
        self.assertAlmostEqual(
            result[-1], groups.group_ret.iloc[-1] - groups.group_ret.iloc[0]
        )
        self.assertTrue((np.unique(factor_groups(x), return_counts=True)[1] >= 2).all())
        rng = np.random.default_rng(17)
        order = rng.permutation(len(x))
        np.testing.assert_array_equal(factor_groups(x[order]), factor_groups(x)[order])
        self.assertTrue(np.isnan(group_metrics(np.ones(100), r, 50)[-1]))

    def test_stock_chunk_state(self):
        cfg = BuildConfig(factors=("a", "b"))
        raw = np.random.default_rng(17).normal(size=(60, 80, 2))
        raw[0, :40, 0] = np.nan
        raw[1, 20:50, 1] = np.nan
        eligible = np.ones((60, 80), bool)
        full = stock_block(raw, eligible, cfg)[0]
        state = {}
        chunks = [
            stock_block(raw[:, a:b], eligible[:, a:b], cfg, state)[0]
            for a, b in [(0, 30), (30, 60), (60, 80)]
        ]
        np.testing.assert_allclose(full, np.concatenate(chunks, axis=1))

    def test_ff3_partial_and_degenerate(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(40, 3))
        y = 2 + x @ np.array([1.0, 2.0, 3.0])
        y[:7] = np.nan
        bad = np.full(40, np.nan)
        bad[:4] = 1
        parts = [
            pd.DataFrame({name: x[:, i]}).rename_axis("date")
            for i, name in enumerate(["mkt", "hml", "smb"])
        ]
        fit = FF3_Model(
            pd.DataFrame({"good": y, "short": bad, "zero": np.zeros(40)}),
            mkt=parts[0], hml=parts[1], smb=parts[2],
        ).fit()
        np.testing.assert_allclose(fit.coef[:, 0], [2, 1, 2, 3], atol=1e-12)
        self.assertEqual(fit.fit_observations["good"], 33)
        self.assertTrue(fit.r2[["short", "zero"]].isna().all())
        self.assertTrue(fit.resid_mom()[["short", "zero"]].isna().all())

    def test_event_source_gaps_and_prefix(self):
        dates = np.arange(400)
        close = np.arange(400, dtype=float) + 100
        quote = pd.DataFrame(
            {
                "trade_date": dates,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "pct_chg": 0.0,
            }
        )
        industry = pd.DataFrame({"trade_date": dates, "close": close})
        full = high_level_switch.from_quotes(
            quote, {"a": industry, "b": industry}, dates
        )
        prefix = high_level_switch.from_quotes(
            quote.iloc[:350],
            {"a": industry.iloc[:350], "b": industry.iloc[:350]},
            dates[:350],
        )
        pd.testing.assert_frame_equal(full.iloc[:350], prefix)
        broken = industry.drop(index=300)
        gap = high_level_switch.from_quotes(quote, {"a": broken, "b": industry}, dates)
        self.assertTrue(pd.isna(gap.loc[300, "high_level_switch"]))
        self.assertEqual(gap.loc[300, "source_reason"], "industry_missing")
        self.assertEqual(full.loc[300, "source_valid"], 1)
        self.assertTrue(full.loc[:242, "high_level_switch"].isna().all())

    def test_event_age_resets(self):
        frame = pd.DataFrame({"high_level_switch": [1.0, 0.0, np.nan, 0.0, 1.0]})
        values, names = market_block(frame, BuildConfig())
        np.testing.assert_array_equal(
            values[0, :, 0, names.index("high_level_switch__age__valid")],
            [1, 1, 0, 0, 1],
        )

    def test_checkpoint_resume_failure_and_revision(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "output"
            cp = Checkpoint(root, {"version": 1})
            cp.step("one", [out], lambda: out.write_text("first"))
            self.assertFalse(
                cp.step("one", [out], lambda: self.fail("repeated completed step"))
            )
            with self.assertRaises(RuntimeError):
                cp.step(
                    "two", [out], lambda: (_ for _ in ()).throw(RuntimeError("test"))
                )
            cp.db.close()
            cp = Checkpoint(root, {"version": 1})
            cp.step("two", [out], lambda: out.write_text("second"))
            cp.db.close()
            with self.assertRaises(ValueError):
                Checkpoint(root, {"version": 2})
            out.write_text("external revision")
            with self.assertRaises(ValueError):
                Checkpoint(root, {"version": 1})

    def test_pool_dependencies(self):
        registry = {
            "one": SimpleNamespace(meta_type="pooling", sub_factors=["-a"]),
            "two": SimpleNamespace(meta_type="pooling", sub_factors=["one"]),
            "other": SimpleNamespace(meta_type="pooling", sub_factors=["z"]),
        }
        self.assertEqual(dependency_order(registry, ["a"]), ["one", "two"])

    def test_market_overwrite(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "event"
            path.touch()
            old = pd.DataFrame({"date": [1, 2], "event": [0.0, 0.0]})
            calc = Mock(spec=["db_src", "db_key", "indent", "validate_value"],
                        db_src="market_factor", db_key="event", indent=0,
                        side_effect=lambda date: pd.DataFrame({"date": [date], "event": [1.0]}))
            calc.validate_value.side_effect = lambda frame, date, strict=False: frame

            with (
                patch(
                    "src.res.factor.calculator.factor_calc.DB.path", return_value=path
                ),
                patch(
                    "src.res.factor.calculator.factor_calc.DB.load", return_value=old
                ),
                patch(
                    "src.res.factor.calculator.factor_calc.DB.save", return_value=True
                ) as save,
            ):
                self.assertFalse(
                    MarketFactorCalculator.calc_and_deploy(calc, 2, overwrite=False)
                )
                MarketFactorCalculator.calc_and_deploy(calc, 2, overwrite=True)
                self.assertEqual(
                    save.call_args.args[0].set_index("date").loc[2, "event"], 1
                )

    def test_repaired_masked_loader(self):
        from src.data.preprocess.factor_sequence.repaired_bundle import REPAIR_VERSION
        from src.data.preprocess.factor_sequence.processors import load_block
        from src.data.preprocess.factor_sequence.bundle import file_digest

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "dates.npy", np.array([20200101, 20200102]))
            np.save(root / "secids.npy", np.array([1, 2]))
            np.save(
                root / "factorperf.npy",
                np.array([[[[0.0, 0.0]], [[0.5, 1.0]]]], dtype="float32"),
            )
            manifest = dict(
                format=REPAIR_VERSION,
                config={"factors": ["a"]},
                blocks={"factorperf": {"features": ["a__ic", "a__ic__valid"]}},
                files={p.name: file_digest(p) for p in root.glob("*.npy")},
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            block = load_block(
                root, "factorperf", dates=[20200102, 20200103], view="slim"
            )
            self.assertEqual(block.values.shape, (1, 2, 1, 2))
            self.assertEqual(block.values[0, 0, 0, 1], 1)
            self.assertEqual(block.values[0, 1, 0, 1], 0)


if __name__ == "__main__":
    unittest.main()
