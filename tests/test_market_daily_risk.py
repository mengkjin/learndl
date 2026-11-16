import tempfile
import importlib
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.update.custom.market_daily_risk import (
    pandas_market_returns,
    polars_minutes,
)
from src.data.update.custom.market_daily_risk import (
    calc_market_daily_risk,
    MarketDailyRiskUpdater,
)

rebuild_script = importlib.import_module("scripts.8_fixes.5_rebuild_market_daily_risk")
validate_day = rebuild_script.validate_day
digest = rebuild_script.digest
replace_checked = rebuild_script.replace_checked
rebuild = rebuild_script.rebuild


class MarketRiskTests(unittest.TestCase):
    def data(self, counts=True):
        q = pd.DataFrame(
            dict(
                secid=[1, 2],
                preclose=[10.0, 100.0],
                high=[12.0, 110.0],
                low=[9.0, 95.0],
                turn_fl=[2.0, 3.0],
                amount=[1000.0, 2000.0],
                vwap=[10.0, 100.0],
            )
        ).set_index("secid")
        v = pd.DataFrame(dict(secid=[1, 2], float_share=[10.0, 20.0])).set_index(
            "secid"
        )
        mf = pd.DataFrame(
            {
                f"{s}_{k}_amount": [10.0, 20.0]
                for s in ("buy", "sell")
                for k in ("sm", "md", "lg", "elg")
            },
            index=q.index,
        )
        mf["buy_lg_vol"] = 1.0
        mf["buy_elg_vol"] = 2.0
        m = (
            pd.DataFrame(
                [
                    dict(
                        secid=s,
                        date=20200102,
                        minute=t,
                        close=p,
                        amount=100.0,
                        num_trades=5.0,
                    )
                    for s, prices in [
                        (1, [10.0, 11.0, 10.0, 12.0, 11.0]),
                        (2, [100.0, 102.0, 110.0, 108.0, 109.0]),
                    ]
                    for t, p in zip([0, 1, 2, 5, 42], prices)
                ]
            )
            .sample(frac=1, random_state=7)
            .set_index("secid")
        )
        if not counts:
            m = m.drop(columns="num_trades")
        return dict(quote=q, val=v, moneyflow=mf, min=m)

    def test_grouped_shift_and_independent_engines(self):
        for counts in (True, False):
            x = self.data(counts)
            ret = pandas_market_returns(x["quote"], x["val"], x["min"])
            self.assertAlmostEqual(ret.loc[1], (0.1 * 100 + 0.02 * 2000) / 2100)
            result = calc_market_daily_risk(20200102, inputs=x)
            self.assertEqual(len(validate_day(x, result.iloc[0])), 4)
            y = x.copy()
            y["min"] = x["min"].iloc[::-1]
            pd.testing.assert_series_equal(
                ret, pandas_market_returns(y["quote"], y["val"], y["min"])
            )

    def test_updater_backend_interface(self):
        x = self.data()
        pandas_result = MarketDailyRiskUpdater.calculate(
            20200102, backend="pandas", inputs=x
        )
        polars_result = MarketDailyRiskUpdater.calculate(
            20200102, backend="polars", inputs=x
        )
        pd.testing.assert_frame_equal(
            pandas_result, polars_result, rtol=1e-9, atol=1e-12
        )
        with self.assertRaises(ValueError):
            MarketDailyRiskUpdater.calculate(20200102, backend="invalid", inputs=x)

    def test_missing_returns_weights_and_constant(self):
        x = self.data()
        x["min"]["close"] = 10.0
        x["min"].iloc[2, list(x["min"].columns).index("close")] = np.nan
        x["val"].iloc[1, 0] = np.nan
        result = calc_market_daily_risk(20200102, inputs=x)
        validate_day(x, result.iloc[0])
        self.assertEqual(result.market_day_5min_ret_skewness.iloc[0], 0.0)

    def test_reject_duplicates_and_multiple_dates(self):
        x = self.data()
        m = pd.concat([x["min"], x["min"].iloc[:1]])
        with self.assertRaises(ValueError):
            pandas_market_returns(x["quote"], x["val"], m)
        x["min"].iloc[0, list(x["min"].columns).index("date")] = 20200103
        with self.assertRaises(ValueError):
            polars_minutes(x["quote"], x["val"], x["min"], x["moneyflow"])

    def test_atomic_replace_checks_concurrent_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, stage = Path(tmp) / "risk.csv", Path(tmp) / "new.csv"
            target.write_text("old")
            stage.write_text("new")
            old = digest(target)
            target.write_text("changed")
            with self.assertRaises(RuntimeError):
                replace_checked(target, stage, old)
            self.assertEqual(target.read_text(), "changed")
            replace_checked(target, stage, digest(target))
            self.assertEqual(target.read_text(), "new")


class RebuildIntegrationTests(unittest.TestCase):
    data = MarketRiskTests.data

    def test_validated_replacement_and_failure_preserves_database(self):
        from unittest.mock import patch
        from contextlib import ExitStack

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "risk.feather"
            old = calc_market_daily_risk(20200101, inputs=self.data())
            old.to_feather(target)
            with ExitStack() as stack:
                stack.enter_context(patch("src.proj.PATH.result", root / "results"))
                stack.enter_context(patch("src.proj.DB.path", return_value=target))
                stack.enter_context(
                    patch(
                        "src.proj.DB.load",
                        side_effect=lambda *a, **kw: pd.read_feather(target),
                    )
                )
                stack.enter_context(
                    patch("src.proj.CALENDAR.range", return_value=[20200102])
                )
                source = stack.enter_context(
                    patch(
                        "src.data.update.custom.market_daily_risk._get_inputs",
                        return_value=self.data(),
                    )
                )
                stack.enter_context(
                    patch("src.automation.artifact_exchange.research.register")
                )
                folder = rebuild(20200102, 20200102)
                self.assertEqual(
                    pd.read_feather(target).date.tolist(), [20200101, 20200102]
                )
                self.assertTrue((folder / "backup/risk.feather").exists())
                before = digest(target)
                source.side_effect = ValueError("missing minute file")
                with self.assertRaisesRegex(ValueError, "missing minute"):
                    rebuild(20200102, 20200102)
                self.assertEqual(digest(target), before)


if __name__ == "__main__":
    unittest.main()
