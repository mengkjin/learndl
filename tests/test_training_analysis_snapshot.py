import csv
import gzip
import json
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

        def source(ds, *, universe_files=None):
            file = "/db/history/day." + str(ds[0]) + ".feather"
            return {file: [file, 10, revision[ds[0] // 10000], 1]}

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
            self.assertEqual(second["reused_years"], [2024])
            self.assertIn("此前没有", second["rebuild_reasons"][2025][0])
            old = root / "generations" / first["generation"] / "2024/market.feather"
            new = root / "generations" / second["generation"] / "2024/market.feather"
            original_inode = old.stat().st_ino
            self.assertEqual(original_inode, new.stat().st_ino)
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
            self.assertEqual(repaired["recomputed_years"], [2024])
            self.assertEqual(repaired["rewritten_years"], [])
            repaired_path = root / "generations" / repaired["generation"] / "2024/market.feather"
            self.assertEqual(original_inode, repaired_path.stat().st_ino)
            self.assertEqual(build_snapshot([20240102], root)["recomputed_years"], [])
            self.assertIn("mtime 变化", repaired["rebuild_reasons"][2024][0])
            self.assertIn("day.20240102.feather", repaired["rebuild_reasons"][2024][0])
            third = build_snapshot([20250103], root)
            self.assertIn("新增覆盖日期（20250103）", third["rebuild_reasons"][2025])
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
            self.assertEqual(unchanged["reused_years"], [2024, 2025])
            self.assertEqual(unchanged["rebuild_reasons"], {})
            rebuilt = build_snapshot([20250102], root, resume=False)
            self.assertEqual(rebuilt["rewritten_years"], [2025])
            with TrainingAnalysisSnapshot(root).activate() as reader:
                self.assertEqual(set(reader.meta["years"]), {"2025"})
                self.assertEqual(reader.meta["years"]["2025"]["dates"], [20250102])

    def test_plan_explains_historical_rebuilds_before_computing(self):
        import json
        from src.res.model.util.data import training_analysis_snapshot as snap

        dates = [20240102, 20250102]
        revision = 1
        def source(ds, *, universe_files=None):
            if universe_files is not None:
                universe_files["/db/description.feather"] = [5, revision, 10]
            return {"universe_dependencies": {str(d): str(revision) for d in ds}}

        messages = []
        def market(ds):
            messages.append(f"compute {ds[0]}")
            return pd.DataFrame({"date": ds, "secid": 1})

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(snap, "source_state", side_effect=source),
            patch.object(snap, "market_frame", side_effect=market),
            patch.object(snap, "evaluation_frame", side_effect=lambda ds: pd.DataFrame({"date": ds, "secid": 1})),
            patch.object(snap, "membership_frame", side_effect=lambda ds: pd.DataFrame({"date": ds, "secid": 1})),
            patch.object(snap.DATAVENDOR, "clear_all"),
            patch.object(snap.Logger, "info", side_effect=messages.append),
        ):
            root = Path(tmp)
            first = build_snapshot(dates, root)
            manifest = root / "generations" / first["generation"] / "manifest.json"
            old = json.loads(manifest.read_text())
            old.pop("dependency_version")
            old["years"]["2024"].pop("universe_files")  # Previous manifest format.
            old["schema"] -= 1
            old["contract"]["price_semantics"] = "previous"
            old["years"]["2024"]["maturity_cutoff"] = 20240102
            manifest.write_text(json.dumps(old))
            revision += 1
            messages.clear()
            result = build_snapshot(dates, root)
            first_compute = messages.index("compute 20240102")
            reasons = result["rebuild_reasons"]
            self.assertIn("计划重算年度 [2024, 2025]，复用年度 []", messages[:first_compute])
            for year, entries in reasons.items():
                self.assertTrue(any("快照格式版本" in s for s in entries))
                self.assertIn("首次建立审计基线", entries)
                self.assertTrue(any("price_semantics" in s and "previous" in s for s in entries))
                for reason in entries:
                    self.assertIn(f"{year} 年重算 · {reason}", messages[:first_compute])
            self.assertTrue(any("成熟截止日" in s for s in reasons[2024]))
            self.assertTrue(any("旧版股票池" in s for s in reasons[2024]))
            self.assertTrue(any("20240102" in s and "股票池依赖指纹" in s for s in reasons[2024]))
            self.assertTrue(any("description.feather" in s and "mtime" in s for s in reasons[2025]))

            folder = root / "generations" / result["generation"]
            (folder / "2024/market.feather").write_bytes(b"corrupt")
            (folder / "2025/membership.feather").unlink()
            result = build_snapshot(dates, root)
            self.assertEqual(result["rebuild_reasons"], {
                2024: ["年度文件校验失败 market.feather"], 2025: ["年度文件缺失 membership.feather"],
            })

    def test_weekly_extends_fit_after_snapshot(self):
        from types import SimpleNamespace
        import src.api.pkgs.update as module

        calls = []
        with (
            patch.object(module, "MACHINE", SimpleNamespace(updatable=True, cuda_server=True)),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.weekly_update",
                side_effect=lambda: calls.append("snapshot") or {
                    "rewritten_years": [2025], "recomputed_years": [2025], "reused_years": [2024], "compute_seconds": 2,
                    "save_seconds": 1, "total_seconds": 4, "generation": "test",
                },
            ),
            patch.object(module.Logger, "Paragraph", wraps=module.Logger.Paragraph) as paragraph,
            patch.object(module.Logger, "warning") as warning,
            patch.object(
                module.ModelAPI,
                "prepare_fit_data",
                side_effect=lambda: calls.append("fit"),
            ),
            patch.object(module, "print_update_records"),
        ):
            module.UpdateAPI.weekly()
            paragraph.assert_any_call("Training Analysis Snapshot", 3)
            warning.assert_not_called()
        self.assertEqual(calls, ["snapshot", "fit"])
        with (
            patch.object(module, "MACHINE", SimpleNamespace(updatable=True, cuda_server=True)),
            patch(
                "src.res.model.util.data.training_analysis_snapshot.weekly_update",
                side_effect=RuntimeError("unavailable"),
            ),
            patch.object(module.ModelAPI, "prepare_fit_data") as models,
            patch.object(module, "print_update_records"),
        ):
            module.UpdateAPI.weekly()
            models.assert_called_once()

    def test_source_policy_ignores_routine_reports_and_tracks_cross_year_quotes(self):
        from src.res.model.util.data import training_analysis_snapshot as snap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def path(src, key, date=None):
                return root / src / (f"{key}.{date}.feather" if date else f"{key}.feather")

            with patch.object(snap.DB, "path", side_effect=path), patch.object(snap.DB, "dates", return_value=[]):
                dates = [20241231]
                digest = snap.source_state(dates)
                for key in snap.ROUTINE_INFO:
                    file = path("information_ts", key)
                    file.parent.mkdir(exist_ok=True)
                    file.write_bytes(b"rewritten")
                self.assertEqual(snap.source_state(dates), digest)
                day = path("trade_ts", "day", 20250227)
                day.parent.mkdir()
                day.write_bytes(b"may affect the evaluation security axis")
                revised = snap.source_state(dates)
                self.assertNotEqual(revised[str(day)], digest[str(day)])
                listing = path("information_ts", "list_day_price")
                listing.write_bytes(b"reference prices changed")
                self.assertNotEqual(snap.source_state(dates)[str(listing)], revised[str(listing)])

    def test_historical_diffs_and_delivery(self):
        from src.automation.artifact_exchange import ArtifactSpec, Config, publish_artifacts, receive
        from src.res.model.util.data import training_analysis_snapshot as snap

        dates = [20240102, 20240103]
        updated = 20240103
        revision = 1

        def source(ds, *, universe_files=None):
            key = "/db/trade/day.20240102.feather"
            return {key: [key, 10, revision, 1]}

        def stable(ds):
            # Independent of coverage in this fixture so only evaluation needs replacement.
            return pd.DataFrame({"date": [dates[0]], "secid": [1]})

        def labels(ds):
            revised = revision > 1
            rows = [
                (dates[0], 1, 20240103, True, 0.25 if revised else 0.1),
                (dates[0], 2, 20240104, revised, 0.2 if revised else float("nan")),
                (dates[1], 1, 20240103, True, 0.3 + (1e-10 if revised else 0)),
            ]
            if revised:
                rows += [(20240104, 1, 20240119, False, float("nan"))]
            return pd.DataFrame(rows, columns=[
                "date", "secid", "label_end", "label_mature", "forward_return_10_2_close",
            ])

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(snap, "source_state", side_effect=source),
            patch.object(snap.CALENDAR, "updated", side_effect=lambda: updated),
            patch.object(snap, "market_frame", side_effect=stable),
            patch.object(snap, "membership_frame", side_effect=stable),
            patch.object(snap, "evaluation_frame", side_effect=labels),
            patch.object(snap.DATAVENDOR, "clear_all"),
        ):
            root = Path(tmp).resolve()
            first = snap.build_snapshot(dates, root)
            revision, updated = 2, 20240104
            dates = [*dates, 20240104]
            second = snap.build_snapshot(dates, root)
            self.assertEqual(second["rewritten_years"], [2024])
            self.assertEqual(second["recomputed_years"], [2024])
            self.assertIn("audit_path", second)
            assert "audit_path" in second
            audit = Path(second["audit_path"])
            result = json.loads((audit / "summary.json").read_text())["years"]["2024"]
            self.assertEqual(result["market"]["status"], "equal")
            self.assertEqual(result["membership"]["status"], "equal")
            self.assertEqual(result["evaluation"]["differences"], 4)
            for table in ("market", "membership"):
                old = root / "generations" / first["generation"] / f"2024/{table}.feather"
                new = root / "generations" / second["generation"] / f"2024/{table}.feather"
                self.assertEqual(old.stat().st_ino, new.stat().st_ino)

            shared = root / "share"
            (shared / ".stfolder").mkdir(parents=True)
            publisher = Config(enabled=True, role="publisher", shared_root=shared,
                               state=root / "publisher", results=root / "source", source_host="fixture")
            receiver = Config(enabled=True, shared_root=shared, state=root / "receiver",
                              results=root / "archive")
            sent = publish_artifacts("weekly_update", second["generation"], [
                ArtifactSpec(audit, category=snap.AUDIT_CATEGORY, experiment="weekly_update"),
            ], config=publisher)
            self.assertEqual(sent[0]["status"], "published")
            received = receive(receiver)
            self.assertEqual(received[0]["status"], "received")
            archived = next(receiver.results.rglob("2024-evaluation.csv.gz"))
            with gzip.open(archived, "rt") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(sum(row["classification"] == "label_maturity" for row in rows), 2)
            self.assertEqual(sum(row["classification"] == "new_coverage" for row in rows), 1)
            historical, = [row for row in rows if row["classification"] == "historical"]
            self.assertEqual((historical["date"], historical["secid"]), ("20240102", "1"))
            self.assertEqual((json.loads(historical["old"]), json.loads(historical["new"])), (0.1, 0.25))
            with gzip.open(audit / "dependencies.json.gz", "rt") as file:
                files = json.load(file)["2024"]["files"]
            changed, = [entry for entry in files if entry["path"].endswith("day.20240102.feather")]
            self.assertEqual((changed["old"], changed["new"]), ([10, 1, 1], [10, 2, 1]))

            from src.automation.artifact_exchange import research

            owner = MagicMock()
            with (
                patch.object(snap, "build_snapshot", return_value=second),
                patch.object(snap.DB, "dates", return_value=dates),
                patch.object(research, "CURRENT_TASK", MagicMock(get=MagicMock(return_value=owner))),
            ):
                self.assertEqual(snap.weekly_update(), second)
            spec, = owner.register_artifact.call_args.args
            self.assertEqual((spec.source, spec.run_id), (audit, second["generation"]))

            unavailable = Config(enabled=True, role="publisher", shared_root=root / "offline-share",
                                 state=root / "offline", results=root / "source")
            with patch.object(research.Config, "load", return_value=unavailable):
                delivery = research.publish_safe("weekly_update", second["generation"], [spec])
            self.assertEqual(delivery[0]["status"], "failed")
            self.assertTrue((audit / "summary.json").is_file())
            self.assertEqual(json.loads((root / "current.json").read_text())["generation"], second["generation"])

            revision = 3
            with patch.object(snap.snapshot_audit, "write_summary", side_effect=OSError("full disk")):
                with self.assertRaisesRegex(OSError, "full disk"):
                    snap.build_snapshot(dates, root)
            self.assertEqual(json.loads((root / "current.json").read_text())["generation"], second["generation"])

    def test_table_diff_keeps_historical_structure_and_missingness(self):
        from src.res.model.util.data import snapshot_audit

        writer = MagicMock()
        diff = snapshot_audit.TableDiff(writer, 2024, "membership")
        old = pd.DataFrame({"date": [20240102] * 2, "secid": [1, 2], "flag": [True, False],
                            "old_column": [1, 2], "value": [None, 0.3]})
        new = pd.DataFrame({"date": [20240102] * 2, "secid": [3, 1], "flag": [True, False],
                            "new_column": [1, 2], "value": [None, 0.5]})
        snapshot_audit.compare_tables(old, new, diff, {20240102}, {20240102}, 20240102, 20240102)
        self.assertEqual(diff.count, 8)
        with self.assertRaisesRegex(ValueError, "invalid.*20240102, 1"):
            snapshot_audit.compare_tables(old, pd.concat([new, new]), diff, set(), set(), 0, 0)

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
