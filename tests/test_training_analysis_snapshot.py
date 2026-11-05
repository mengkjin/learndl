from typing import cast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.res.model.util.data.training_analysis_snapshot import (
    build_snapshot,
    TrainingAnalysisSnapshot,
    prune_generations,
)
from src.res.factor.util.classes.analysis_inputs import CURRENT, return_label_scope
from src.res.factor.util.classes.stock_factor import append_fut_ret


class TrainingAnalysisSnapshotTest(unittest.TestCase):
    def test_default_label_and_parameter_fallback(self):
        index = pd.MultiIndex.from_tuples([(1, 20240102)], names=["secid", "date"])
        factor = pd.DataFrame({"alpha": [1.0]}, index=index)
        snapshot = MagicMock()
        snapshot.evaluation.side_effect = lambda kind, ids, dates, params: (
            pd.DataFrame({"ret": [0.1]}, index=index)
            if params == (10, 2, "close")
            else None
        )
        token = CURRENT.set(snapshot)
        try:
            with (
                patch(
                    "src.res.factor.util.classes.stock_factor.DATAVENDOR.nday_fut_ret"
                ) as old,
                return_label_scope(),
            ):
                old.return_value.to_dataframe.return_value = pd.DataFrame(
                    {"ret": [0.2]}, index=index
                )
                self.assertEqual(append_fut_ret(factor).ret.iloc[0], 0.1)
                self.assertEqual(append_fut_ret(factor).ret.iloc[0], 0.1)
                snapshot.evaluation.assert_called_once()
                old.assert_not_called()
                self.assertEqual(append_fut_ret(factor, nday=20).ret.iloc[0], 0.2)
                old.assert_called_once()
        finally:
            CURRENT.reset(token)
        with (
            patch(
                "src.res.model.util.data.training_analysis_snapshot.DATAVENDOR.nday_fut_ret"
            ) as old,
            return_label_scope(),
        ):
            old.return_value.to_dataframe.return_value = pd.DataFrame(
                {"ret": [0.3]}, index=index
            )
            for _ in range(2):
                self.assertEqual(append_fut_ret(factor).ret.iloc[0], 0.3)
            old.assert_called_once()

    def test_increment_revision_publication_and_scope(self):
        revision = {2024: 1, 2025: 1}

        def source(ds):
            return {"version": revision[ds[0] // 10000]}

        def table(ds):
            return pd.DataFrame(
                {
                    "secid": [1] * len(ds),
                    "date": ds,
                    "forward_return_10_2_close": [0.1] * len(ds),
                    "universe_member": [True] * len(ds),
                }
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "src.res.model.util.data.training_analysis_snapshot.source_state",
                side_effect=source,
            ),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.market_frame",
                side_effect=table,
            ),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.evaluation_frame",
                side_effect=table,
            ),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.membership_frame",
                side_effect=table,
            ),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.DATAVENDOR.clear_all"
            ),
        ):
            root = Path(tmp)
            first = build_snapshot([20240102], root)
            second = build_snapshot([20250102], root)
            self.assertEqual(second["rewritten_years"], [2025])
            old = root / "generations" / first["generation"] / "2024/market.feather"
            new = root / "generations" / second["generation"] / "2024/market.feather"
            self.assertEqual(old.stat().st_ino, new.stat().st_ino)
            self.assertEqual(build_snapshot([20250102], root)["rewritten_years"], [])
            import fcntl

            with (root / "writer.lock").open("a") as writer:
                fcntl.flock(writer, fcntl.LOCK_EX)
                with self.assertRaises(BlockingIOError):
                    TrainingAnalysisSnapshot(root, allow_partial=True)

            snapshot = TrainingAnalysisSnapshot(root)
            with snapshot.activate():
                self.assertIs(CURRENT.get(), snapshot)
                result = snapshot.evaluation("ret", [1], [20240102])
                assert result is not None
                self.assertEqual(result.ret.iloc[0], 0.1)
                self.assertIsNone(
                    snapshot.evaluation("ret", [1], [20240102], (20, 2, "close"))
                )
            self.assertIsNone(CURRENT.get())
            with self.assertRaisesRegex(ValueError, "coverage"):
                TrainingAnalysisSnapshot(root, dates=[20240103])
            self.assertIsNone(snapshot.evaluation("ret", [999], [20240102]))
            pinned = TrainingAnalysisSnapshot(root)
            revision[2024] = 2
            with self.assertRaises(ValueError):
                TrainingAnalysisSnapshot(root)
            repaired = build_snapshot([20240102], root)
            self.assertEqual(repaired["rewritten_years"], [2024])
            third = build_snapshot([20250103], root)
            removed = prune_generations(root)
            self.assertNotIn(second["generation"], removed)
            pinned.close()
            self.assertIn(second["generation"], prune_generations(root))
            repaired = third
            with patch(
                "src.res.model.util.data.training_analysis_snapshot.evaluation_frame",
                side_effect=RuntimeError("interrupted"),
            ):
                revision[2024] = 3
                with self.assertRaises(RuntimeError):
                    build_snapshot([20240102], root)
            self.assertIn(repaired["generation"], (root / "current.json").read_text())
            revision[2024] = 2
            with TrainingAnalysisSnapshot(
                root, allow_partial=True
            ).activate() as reader:
                (reader.folder / "2024/evaluation.feather").unlink()
                self.assertIsNone(reader.evaluation("ret", [1], [20240102]))
                self.assertIsNone(reader.evaluation("style", [1], [20240102]))
                self.assertIsNotNone(reader.table("membership", [20240102]))
            orphan = root / "generations" / ("a" * 32)
            orphan.mkdir()
            (orphan / "partial.feather").write_bytes(b"interrupted")
            (root / ("." + "a" * 32 + ".json")).write_text("partial")
            for version in range(3, 9):
                revision[2025] = version
                build_snapshot([20250102], root)
                self.assertLessEqual(len(list((root / "generations").iterdir())), 2)
            self.assertFalse(orphan.exists())
            self.assertFalse(list(root.glob(".*.json")))
            unchanged = build_snapshot([20250102], root)
            self.assertEqual(unchanged["rewritten_years"], [])
            rebuilt = build_snapshot([20250102], root, resume=False)
            self.assertEqual(rebuilt["rewritten_years"], [2025])
            with TrainingAnalysisSnapshot(root).activate() as reader:
                self.assertEqual(set(reader.meta["years"]), {"2025"})
                self.assertEqual(reader.meta["years"]["2025"]["dates"], [20250102])

    def test_weekly_orders_snapshot_before_models(self):
        from types import SimpleNamespace
        import src.api.pkgs.update as module

        calls = []
        with (
            patch.object(module, "MACHINE", SimpleNamespace(updatable=True)),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.weekly_update",
                side_effect=lambda: calls.append("snapshot") or {},
            ),
            patch.object(
                module.ModelAPI,
                "update_models",
                side_effect=lambda: calls.append("models"),
            ),
            patch.object(module, "print_update_records"),
        ):
            module.UpdateAPI.weekly()
        self.assertEqual(calls, ["snapshot", "models"])
        with (
            patch.object(module, "MACHINE", SimpleNamespace(updatable=True)),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.weekly_update",
                side_effect=RuntimeError("unavailable"),
            ),
            patch.object(module.ModelAPI, "update_models") as models,
            patch.object(module, "print_update_records"),
        ):
            module.UpdateAPI.weekly()
            models.assert_called_once()

    def test_callback_auto_fallback_and_required_failure(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from src.res.model.callback.test.detail import DetailedAlphaAnalysis

        callback = SimpleNamespace(
            tasks=["factor"],
            factor_names=["best"],
            test_dates=[20240102],
            analysis_snapshot="auto",
            logger=SimpleNamespace(
                timer=lambda *a: nullcontext(),
                paragraph=lambda *a: nullcontext(),
                note=MagicMock(),
            ),
            factor_test=MagicMock(),
            fmp_test=MagicMock(),
            display_export=MagicMock(),
        )
        with (
            patch(
                "src.res.model.util.data.training_analysis_snapshot.TrainingAnalysisSnapshot",
                side_effect=FileNotFoundError,
            ),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.build_snapshot",
                side_effect=AssertionError("callback must not build a snapshot"),
            ),
        ):
            DetailedAlphaAnalysis.on_test_end(cast(DetailedAlphaAnalysis, callback))
            callback.factor_test.assert_called_once()
            callback.analysis_snapshot = "required"
            with self.assertRaises(OSError):
                DetailedAlphaAnalysis.on_test_end(cast(DetailedAlphaAnalysis, callback))

    def test_callback_uses_complete_membership_and_falls_back_without_warnst(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from src.proj.bases import TestType
        from src.res.model.callback.test.detail import DetailedAlphaAnalysis

        dates = [20240102]
        membership = pd.DataFrame(
            {
                "date": dates * 2,
                "secid": [1, 2],
                "universe_member": [True, True],
                "st": [False, False],
                "lowprice": [False, False],
                "bse": [False, False],
                "loser": [False, False],
                "warnst": [False, True],
            }
        )
        snapshot = object.__new__(TrainingAnalysisSnapshot)
        callback = cast(
            DetailedAlphaAnalysis,
            SimpleNamespace(
                fmp_tasks=[TestType.T50],
                t50_backend="auto",
                prepared_universe_root=None,
                prepared_universe_mode="off",
                get_factor_for_fmp_test=lambda: MagicMock(date=dates),
                logger=MagicMock(
                    subprocess=lambda **kw: nullcontext(),
                    timer=lambda *args, **kw: nullcontext(),
                ),
                snap_folder=None,
                config=SimpleNamespace(is_resuming=False, model_name="test"),
                trainer=SimpleNamespace(
                    config=SimpleNamespace(beg_date=dates[0], end_date=dates[0])
                ),
                indent=0,
                vb_level=0,
                test_results={},
                test_figures={},
            ),
        )
        with (
            snapshot.activate(),
            patch.object(snapshot, "table", return_value=membership) as table,
            patch(
                "src.res.factor.util.classes.universe.Universe",
            ) as loader,
            patch("src.res.model.callback.test.detail.FactorTestAPI.run_test") as run,
        ):
            self.assertIsNone(snapshot.universe(dates, name="all"))
            DetailedAlphaAnalysis.fmp_test(callback)
            loader.assert_not_called()
            portfolio = run.call_args.kwargs["universe_portfolio"]
            self.assertEqual(portfolio.get(dates[0]).secid.tolist(), [1])
            table.return_value = membership.drop(columns="warnst")
            loader.side_effect = lambda *args, **kw: MagicMock()
            DetailedAlphaAnalysis.fmp_test(callback)
            loader.assert_called_once()

    def test_quote_outer_join_keeps_presence_masks(self):
        from src.res.model.util.data.training_analysis_snapshot import market_frame
        from types import SimpleNamespace

        info = pd.DataFrame({"indus": ["a"]}, index=pd.Index([1], name="secid"))
        quotes = [
            pd.DataFrame({"ret": [0.1]}, index=pd.Index([i], name="secid"))
            for i in (1, 2, 3)
        ]
        with patch(
            "src.res.model.util.data.training_analysis_snapshot.DATAVENDOR",
            SimpleNamespace(
                INFO=SimpleNamespace(get_indus=lambda d: info),
                td=lambda d, n: 20240103,
                get_quote_ret=MagicMock(side_effect=quotes),
            ),
        ):
            frame = market_frame([20240102]).set_index("secid")
        for i, price in enumerate(("close", "open", "vwap"), 1):
            mask = frame[f"close_to_next_{price}_present"]
            self.assertEqual(mask.dtype, bool)
            self.assertEqual(list(frame.index[mask]), [i])
