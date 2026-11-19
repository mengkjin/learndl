"""Recovery adoption, scoped IO isolation and daily-update overlap regressions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src.data.preprocess.factor_sequence.bundle import file_digest
from src.data.maintenance.factor_sequence.repair import Checkpoint
from src.data.maintenance.factor_sequence.recovery import RecoverySource, run_recovery
from src.data.preprocess.factor_sequence.repaired_bundle import REPAIR_VERSION
from src.proj.db.io.snapshot import ReadSnapshot
from src.proj.db.io.dataframe import dfIOHandler
from src.proj.db.basic.df_handler import dfHandler, quiet_empty_stock_factors

__all__ = ["RecoveryTests"]


def parent_checkpoint(folder, output):
    identity = dict(
        version=REPAIR_VERSION,
        mode="rebuild",
        start=20110101,
        end=20261113,
        affected_end=20261113,
    )
    checkpoint = Checkpoint(folder, identity)
    checkpoint.step(
        "factor:dtop:20110104", [output], lambda: output.write_text("factor")
    )
    try:
        checkpoint.step(
            "data", [], lambda: (_ for _ in ()).throw(ValueError("raw changed"))
        )
    except ValueError:
        pass
    checkpoint.db.close()
    return identity


class RecoveryTests(unittest.TestCase):
    def test_snapshot_survives_daily_replacement_and_restart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "db"
            raw.mkdir()
            path = raw / "prices.20110104.feather"
            pd.DataFrame({"v": [1.0]}).to_feather(path)
            snapshot = ReadSnapshot(root / "snapshot", raw)
            with snapshot.activate():
                self.assertEqual(dfIOHandler.load_pandas(path).v.iloc[0], 1)
                pd.DataFrame({"v": [9.0]}).to_feather(path)
                self.assertEqual(dfIOHandler.load_polars(path)["v"][0], 1)
                self.assertEqual(
                    dfIOHandler.load_pandas_multiple([path])[20110104].v.iloc[0], 1
                )
                self.assertEqual(
                    dfIOHandler.load_polars_multiple([path], accelerator="lazy")[
                        20110104
                    ]["v"][0],
                    1,
                )
            self.assertEqual(dfIOHandler.load_pandas(path).v.iloc[0], 9)
            snapshot.close()
            resumed = ReadSnapshot(root / "snapshot", raw)
            with resumed.activate():
                self.assertEqual(dfIOHandler.load_pandas(path).v.iloc[0], 1)
            resumed.close()

    def test_snapshot_missing_and_unrelated_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "db"
            raw.mkdir()
            missing = raw / "missing.feather"
            output = root / "factor.feather"
            snapshot = ReadSnapshot(root / "snapshot", raw)
            with snapshot.activate():
                self.assertTrue(dfIOHandler.load_pandas(missing).empty)
                pd.DataFrame({"v": [2]}).to_feather(missing)
                self.assertTrue(dfIOHandler.load_pandas(missing).empty)
                pd.DataFrame({"v": [3]}).to_feather(output)
                self.assertEqual(dfIOHandler.load_pandas(output).v.iloc[0], 3)
                pd.DataFrame({"v": [4]}).to_feather(output)
                self.assertEqual(dfIOHandler.load_pandas(output).v.iloc[0], 4)
            self.assertEqual(snapshot.summary()["missing"], 1)
            snapshot.close()

    def test_copy_retries_a_concurrent_replacement(self):
        import shutil

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "db"
            raw.mkdir()
            source = raw / "source.feather"
            source.write_bytes(b"old")
            snapshot = ReadSnapshot(root / "snapshot", raw)
            copy = shutil.copyfile
            count = 0

            def replacing_copy(src, dst):
                nonlocal count
                result = copy(src, dst)
                count += 1
                if count == 1:
                    source.write_bytes(b"new source content")
                return result

            with patch(
                "src.proj.db.io.snapshot.shutil.copyfile", side_effect=replacing_copy
            ):
                target = Path(snapshot.resolve(source))
            self.assertEqual(target.read_bytes(), b"new source content")
            self.assertEqual(count, 2)
            snapshot.close()

    def test_corrupt_snapshot_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "db"
            raw.mkdir()
            path = raw / "x.feather"
            path.write_bytes(b"original")
            snapshot = ReadSnapshot(root / "snapshot", raw)
            target = Path(snapshot.resolve(path))
            snapshot.close()
            target.write_bytes(b"corrupt")
            resumed = ReadSnapshot(root / "snapshot", raw)
            with self.assertRaisesRegex(ValueError, "Corrupt private"):
                resumed.resolve(path)
            resumed.close()

    def test_only_verified_factor_artifacts_are_adopted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "factor"
            parent_checkpoint(root / "old", output)
            before = file_digest(root / "old/checkpoint.sqlite")
            source = RecoverySource(root / "old")
            child = Checkpoint(root / "new", {"new": True})
            self.assertTrue(source.adopt("factor:dtop:20110104", output, child))
            self.assertFalse((root / "new/backups").exists())
            self.assertFalse(source.adopt("data", output, child))
            output.write_text("changed")
            self.assertFalse(source.adopt("factor:dtop:20110104", output, child))
            child.db.close()
            source.close()
            self.assertEqual(before, file_digest(root / "old/checkpoint.sqlite"))

    def test_wrong_parent_stage_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_checkpoint(root / "old", root / "factor")
            import sqlite3

            with sqlite3.connect(root / "old/checkpoint.sqlite") as db:
                db.execute("update steps set key='event' where key='data'")
            with self.assertRaisesRegex(ValueError, "only its data step"):
                RecoverySource(root / "old")

    def test_empty_stock_messages_are_scoped_and_counted(self):
        with patch("src.proj.db.basic.df_handler.Logger.only_once") as message:
            with quiet_empty_stock_factors() as counts:
                dfHandler.load_process_pandas(
                    pd.DataFrame(), syntax="stock_factor.dtop.20110104"
                )
                import polars as pl

                dfHandler.load_process_polars(
                    pl.DataFrame(), syntax="stock_factor.ff_r2_1m.20110104"
                )
                message.assert_not_called()
                dfHandler.load_process_pandas(
                    pd.DataFrame(), syntax="trade_ts.day.20110104"
                )
                self.assertEqual(message.call_count, 1)
            self.assertEqual(sum(counts.values()), 2)
            dfHandler.load_process_pandas(
                pd.DataFrame(), syntax="stock_factor.dtop.20110105"
            )
            self.assertEqual(message.call_count, 2)

    def test_recovery_wrapper_uses_snapshot_and_preserves_cutoff(self):
        from src.proj import DB

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "results/factor_sequence_fixes/old"
            parent_checkpoint(parent, root / "factor")
            raw = root / "db"
            raw.mkdir()
            original = raw / "source.feather"
            pd.DataFrame({"v": [1]}).to_feather(original)
            path = SimpleNamespace(result=root / "results", db=raw)

            def run(
                mode, start, end, run_id, dry_run, batch_size, *, recovery, snapshot
            ):
                self.assertEqual(mode, "recover")
                self.assertEqual(recovery.identity["affected_end"], 20261113)
                self.assertEqual(dfIOHandler.load_pandas(original).v.iloc[0], 1)
                pd.DataFrame({"v": [2]}).to_feather(original)
                self.assertEqual(dfIOHandler.load_pandas(original).v.iloc[0], 1)
                self.assertIsNotNone(snapshot)
                return root / run_id

            with (
                patch("src.proj.PATH", path),
                patch(
                    "src.data.maintenance.factor_sequence.repair.run_repair",
                    side_effect=run,
                ),
                patch("src.data.DATAVENDOR"),
            ):
                self.assertEqual(
                    run_recovery(run_id="new", source_run="old"), root / "new"
                )
            report = json.loads(
                (
                    root / "results/factor_sequence_fixes/new/reports/raw_snapshot.json"
                ).read_text()
            )
            self.assertEqual(report["files"], 1)
            self.assertIsNotNone(DB)

    def test_engine_backfills_then_refreshes_before_build_without_global_inventory(
        self,
    ):
        from contextlib import ExitStack
        import numpy as np
        from src.data.maintenance.factor_sequence.repair import run_repair
        import importlib

        for module in (
            "src.data",
            "src.res.factor.calculator",
            "src.res.factor.defs.stock.level0.behavior.behavior_ff3",
            "src.automation.artifact_exchange.research",
        ):
            importlib.import_module(module)

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            dates = [20110104, 20110105]
            raw = root / "db"
            raw.mkdir()
            raw_path = raw / "input.feather"
            pd.DataFrame({"v": [1]}).to_feather(raw_path)
            calls = []

            def path(src, key, date=None):
                return root / "exports" / src / f"{key}.{date}.feather"

            def save(frame, src, key, date=None):
                target = path(src, key, date)
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.to_feather(target)
                return True

            def load(src, key, date=None, **kwargs):
                target = path(src, key, date)
                return pd.read_feather(target) if target.exists() else pd.DataFrame()

            def values(name, date):
                result = pd.DataFrame({"secid": [1, 2], "date": date, name: [1.0, 2.0]})
                if name == "dtop":
                    result["dtop__source_observed"] = 1
                    result["dtop__source_filled"] = 0
                return result

            def calculator(name):
                class Factor:
                    init_date = 20110101
                    db_src = "stock_factor"
                    db_key = name
                    factor_name = name
                    meta_type = "stock"

                    def calc_and_deploy(self, date, **kwargs):
                        calls.append((name, date))
                        self_test.assertEqual(
                            dfIOHandler.load_pandas(raw_path).v.iloc[0], 1
                        )
                        # A daily update replaces a raw table during this run.
                        pd.DataFrame({"v": [9]}).to_feather(raw_path)
                        return save(values(name, date), self.db_src, name, date)

                    def stats_target_dates(self, *args, **kwargs):
                        return {"daily": np.array(dates), "weekly": np.array(dates)}

                    def stock_factor(self, subset, **kwargs):
                        self_test.assertTrue(
                            all(path("stock_factor", name, d).exists() for d in dates)
                        )
                        frame = pd.DataFrame({"date": subset, "metric": 1.0})
                        return SimpleNamespace(
                            daily_stats=lambda: frame, weekly_stats=lambda: frame
                        )

                return Factor

            self_test = self
            event_frame = pd.DataFrame(
                {
                    "date": dates,
                    "source_valid": 1,
                    "source_reason": "valid",
                    "industry_required": 1,
                    "industry_available": 1,
                    "high_level_switch": 0,
                }
            )

            class Event:
                db_src = "market_factor"
                db_key = "high_level_switch"
                meta_type = "market"

                def calc_history(self, end):
                    return event_frame.copy()

                def validate_value(self, value, *args, **kwargs):
                    return value

            pool = calculator("pool")
            pool.meta_type = "pooling"
            setattr(pool, "sub_factors", ["x"])
            setattr(
                pool,
                "calc_pooling_weight",
                lambda self, dates, overwrite: pd.DataFrame(
                    {"date": dates, "weight": 1.0}
                ),
            )
            registry = {
                "pool": pool,
                "dtop": calculator("dtop"),
                "x": calculator("x"),
                "high_level_switch": Event,
            }
            old = root / "old"
            original = path("stock_factor", "dtop", dates[0])
            original.parent.mkdir(parents=True)
            identity = dict(
                version=REPAIR_VERSION,
                mode="rebuild",
                start=20110101,
                end=20110105,
                affected_end=20110105,
            )
            parent = Checkpoint(old, identity)
            parent.step(
                "factor:dtop:20110104",
                [original],
                lambda: save(
                    values("dtop", dates[0]), "stock_factor", "dtop", dates[0]
                ),
            )
            parent.db.execute(
                "insert into steps values ('data','failed','raw changed')"
            )
            parent.db.commit()
            parent.db.close()
            save(values("x", dates[1]), "stock_factor", "x", dates[1])
            source = RecoverySource(old)
            snapshot = ReadSnapshot(root / "snapshot", raw)
            stack.callback(source.close)
            stack.callback(snapshot.close)
            stack.enter_context(snapshot.activate())
            fake_path = SimpleNamespace(
                result=root / "results",
                db=raw,
                srcs=root / "src",
                interim=root / "interim",
            )
            stack.enter_context(patch("src.proj.PATH", fake_path))
            stack.enter_context(
                patch("src.proj.DB", SimpleNamespace(path=path, load=load, save=save))
            )
            stack.enter_context(
                patch(
                    "src.proj.CALENDAR",
                    SimpleNamespace(
                        updated=lambda: 20261117, range=lambda *args: dates
                    ),
                )
            )
            stack.enter_context(patch("src.data.DATAVENDOR"))
            stack.enter_context(
                patch(
                    "src.res.factor.calculator.StockFactorHierarchy",
                    return_value=SimpleNamespace(pool=registry),
                )
            )
            stack.enter_context(
                patch(
                    "src.data.maintenance.factor_sequence.repair.FACTORS", ("dtop", "x")
                )
            )
            inventory = stack.enter_context(
                patch(
                    "src.data.maintenance.factor_sequence.repair._inventory",
                    side_effect=AssertionError("Must not scan live DB"),
                )
            )
            stack.enter_context(
                patch("src.automation.artifact_exchange.research.register")
            )
            stack.enter_context(
                patch(
                    "src.data.preprocess.factor_sequence.repaired_bundle.load_repaired_manifest"
                )
            )

            def build(directory, start, end, batch_size, **kwargs):
                self.assertEqual(end, 20110105)
                for name in ("dtop", "x", "pool"):
                    self.assertTrue(path("factor_stats_daily", name).exists())
                    self.assertTrue(path("factor_stats_weekly", name).exists())
                self.assertEqual(dfIOHandler.load_pandas(raw_path).v.iloc[0], 1)
                self.assertTrue(
                    kwargs["input_provenance"]()["inherited_raw_history_unverified"]
                )
                return root / "bundle"

            builder = stack.enter_context(
                patch(
                    "src.data.maintenance.factor_sequence.repair.build_repaired_project",
                    side_effect=build,
                )
            )
            folder = run_repair(
                "recover",
                end=20110105,
                run_id="new",
                recovery=source,
                snapshot=snapshot,
            )
            self.assertEqual(
                set(calls),
                {
                    ("dtop", dates[1]),
                    ("x", dates[0]),
                    ("pool", dates[0]),
                    ("pool", dates[1]),
                },
            )
            self.assertTrue(path("pooling_weight", "pool").exists())
            inventory.assert_not_called()
            builder.assert_called_once()
            self.assertEqual(
                json.loads((folder / "reports/summary.json").read_text())["status"],
                "completed",
            )
            calls.clear()
            run_repair(
                "recover",
                end=20110105,
                run_id="new",
                recovery=source,
                snapshot=snapshot,
            )
            self.assertEqual(calls, [])
            builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
