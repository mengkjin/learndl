from typing import cast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from src.res.factor.util.classes.prepared_universe import PreparedUniverse
from src.res.factor.util.classes.universe import Universe


class PreparedUniverseTest(unittest.TestCase):
    @patch.object(
        PreparedUniverse,
        "dependencies",
        lambda self, dates: {str(d): "fixture" for d in dates},
    )
    def test_roundtrip_revision_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.feather"
            df = pd.DataFrame(dict(secid=[1, 2], date=[20240102] * 2, st=[False, True]))
            df.to_feather(source)
            with patch(
                "src.res.factor.util.classes.prepared_universe.DB.path",
                return_value=source,
            ):
                store = PreparedUniverse(root / "packs", "test")
                store.prepare([20240102])
                pd.testing.assert_frame_equal(store.load([20240102], "st"), df)
                with self.assertRaises(ValueError):
                    store.load([20240102], "st_warnst")
                with self.assertRaises(ValueError):
                    store.load([20240103], "st")
                df.assign(st=False).to_feather(source)
                with self.assertRaises(ValueError):
                    store.load([20240102], "st")
                store.prepare([20240102])
                self.assertFalse(store.load([20240102], "st").st.any())

    def test_missing_columns_are_saved(self):
        df = pd.DataFrame(dict(secid=[1], date=[20240102]))
        with (
            patch("src.res.factor.util.classes.universe.DB.path") as path,
            patch("src.res.factor.util.classes.universe.Load.df", return_value=df),
            patch(
                "src.res.factor.util.classes.universe.UniverseExclusions.get_exclusion_list",
                return_value=[],
            ),
            patch("src.res.factor.util.classes.universe.Save.df") as save,
        ):
            path.return_value.exists.return_value = True
            Universe("test").get_universe_df(20240102)
            save.assert_called_once()

    def test_incremental_and_upstream_revision(self):
        from src.res.factor.util.classes.prepared_universe import StaleUniverse

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = [20240102, 20240201, 20240202]
            for date in dates:
                pd.DataFrame(dict(secid=[1], date=[date], st=[False])).to_feather(
                    root / f"{date}.feather"
                )
            deps = {str(d): "initial" for d in dates}
            with (
                patch(
                    "src.res.factor.util.classes.prepared_universe.DB.path",
                    side_effect=lambda src, key, date: root / f"{date}.feather",
                ),
                patch.object(
                    PreparedUniverse,
                    "dependencies",
                    side_effect=lambda ds: {str(d): deps[str(d)] for d in ds},
                ),
            ):
                store = PreparedUniverse(root / "packs", "top-1000")
                first = store.prepare(dates[:2])
                old, _ = store.current()
                second = store.prepare(dates)
                current, _ = store.current()
                assert old is not None and current is not None
                self.assertEqual(second["changed_dates"], 1)
                self.assertEqual(second["rewritten_months"], 1)
                self.assertEqual(
                    (old / "202401.feather").stat().st_ino,
                    (current / "202401.feather").stat().st_ino,
                )
                self.assertEqual(len(store.load(dates, "st")), 3)
                self.assertEqual(
                    store.prepare(dates)["generation"], second["generation"]
                )
                deps[str(dates[0])] = "revised"
                with self.assertRaises(StaleUniverse):
                    store.load(dates, "st")
                with self.assertRaises(StaleUniverse):
                    store.prepare(dates)
                _, manifest = store.current()
                assert manifest is not None
                self.assertEqual(manifest["generation"], second["generation"])
                self.assertNotEqual(first["generation"], second["generation"])
                self.assertEqual(len(store.load(dates[1:], "st")), 2)

    def test_dependency_window_and_late_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def path(src, key, date=None):
                return root / f"{src}.{key}.{date}"

            with (
                patch(
                    "src.res.factor.util.classes.prepared_universe.DB.path",
                    side_effect=path,
                ),
                patch(
                    "src.res.factor.util.classes.prepared_universe.DB.dates",
                    return_value=[],
                ),
            ):
                store = PreparedUniverse(root / "packs", "top-1000")
                dates = [20240102, 20240502]
                before = store.dependencies(dates)
                path("crawler", "announcement_sse", 20240101).write_text("late arrival")
                after = store.dependencies(dates)
                self.assertNotEqual(before["20240102"], after["20240102"])
                self.assertEqual(before["20240502"], after["20240502"])
                path("information_ts", "description").write_text("revised info")
                final = store.dependencies(dates)
                self.assertNotEqual(after["20240502"], final["20240502"])


class CallbackUniverseReuseTest(unittest.TestCase):
    def test_callback_constructs_once_for_four_tasks(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.proj.bases import TestType
        from src.res.model.callback.test.detail import DetailedAlphaAnalysis

        factor = MagicMock()
        factor.date = [20240102]
        logger = MagicMock()
        logger.subprocess.side_effect = lambda **kw: nullcontext()
        logger.timer.side_effect = lambda *a, **kw: nullcontext()
        cb = SimpleNamespace(
            fmp_tasks=[
                TestType.T50,
                TestType.SCREEN,
                TestType.REVSCREEN,
                TestType.REINFORCE,
            ],
            logger=logger,
            get_factor_for_fmp_test=lambda: factor,
            prepared_universe_root=None,
            prepared_universe_mode="off",
            snap_folder=Path("/tmp"),
            config=SimpleNamespace(is_resuming=False, model_name="test"),
            trainer=SimpleNamespace(
                config=SimpleNamespace(beg_date=20240102, end_date=20240102)
            ),
            indent=0,
            vb_level=1,
            t50_backend="auto",
            test_results={},
            test_figures={},
        )
        with (
            patch("src.res.factor.util.classes.universe.Universe") as universe,
            patch("src.res.model.callback.test.detail.FactorTestAPI.run_test") as run,
        ):
            run.return_value.get_rslts.return_value = {}
            run.return_value.get_figs.return_value = {}
            DetailedAlphaAnalysis.fmp_test(cast(DetailedAlphaAnalysis, cb))
            universe.return_value.get.assert_called_once_with(factor.date)
            self.assertEqual(run.call_count, 4)
            supplied = [
                call.kwargs["universe_portfolio"] for call in run.call_args_list
            ]
            self.assertTrue(all(port is supplied[0] for port in supplied))

    def test_generators_consume_supplied_portfolio(self):
        import importlib
        from unittest.mock import MagicMock

        for kind, clsname in [
            ("t50", "T50FMPTest"),
            ("screen", "ScreenFMPTest"),
            ("revscreen", "RevScreenFMPTest"),
            ("reinforce", "ReinforceFMPTest"),
        ]:
            module = importlib.import_module(f"src.res.factor.analytic.fmp_{kind}")
            # Class naming is resolved by TEST_TYPE, not the display spelling.
            cls = next(
                v
                for k, v in vars(module).items()
                if isinstance(v, type) and k.endswith("FMPTest")
            )
            task = MagicMock()
            portfolio = MagicMock()
            task.kwargs = {"universe_portfolio": portfolio}
            factor = MagicMock()
            with (
                patch.object(module, "PortfolioGroupBuilder") as builder,
                patch("src.res.factor.fmp.generator_array.run_generator"),
                patch("src.res.factor.util.classes.universe.Universe") as universe,
            ):
                cls.generate(task, factor)
                self.assertIs(builder.call_args.args[2][0], portfolio)
                self.assertNotIn("universe_portfolio", builder.call_args.kwargs)
                universe.assert_not_called()

    def test_repair_rebuilds_revised_day_and_keeps_other_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = [20240102, 20240202]
            for d in dates:
                pd.DataFrame(dict(secid=[1], date=[d], st=[False])).to_feather(
                    root / f"{d}.feather"
                )
            deps = {str(d): "original" for d in dates}

            def rebuild(self, date, **kw):
                self_test.assertTrue(kw["rebuild"])
                pd.DataFrame(dict(secid=[1], date=[date], st=[True])).to_feather(
                    root / f"{date}.feather"
                )

            self_test = self
            with (
                patch(
                    "src.res.factor.util.classes.prepared_universe.DB.path",
                    side_effect=lambda src, key, date: root / f"{date}.feather",
                ),
                patch.object(
                    PreparedUniverse,
                    "dependencies",
                    side_effect=lambda ds: {str(d): deps[str(d)] for d in ds},
                ),
                patch.object(
                    Universe, "get_universe_df", autospec=True, side_effect=rebuild
                ) as build,
                patch("src.data.DATAVENDOR.INFO.clear_all"),
                patch("src.data.DATAVENDOR.TRADE.clear_all"),
                patch(
                    "src.res.factor.util.classes.universe.UniverseExclusions.prepare_warnst"
                ),
                patch("src.proj.Save.async_wait_all"),
            ):
                store = PreparedUniverse(root / "packs", "top-1000")
                store.prepare(dates)
                deps[str(dates[0])] = "changed"
                with self.assertRaises(ValueError):
                    store.prepare_from_sources(dates)
                build.assert_not_called()
                report = store.prepare_from_sources(
                    dates, verified_warning_history=True
                )
                self.assertEqual(report["changed_dates"], 1)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(store.load(dates, "st").st.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
