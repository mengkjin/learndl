"""Tests for factor-sequence progress tracking and recovery decisions."""
from __future__ import annotations

__all__ = [
    'ProgressTests',
]
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from src.automation.scheduling.factor_sequence_progress import collect


class ProgressTests(unittest.TestCase):
    def test_missing_ledger_is_not_created(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = collect(root / "results", root / "runtime", "test", isolated=True)
            self.assertIn("scheduler_read_error", report)
            self.assertFalse((root / "runtime").exists())

    def test_running_is_not_claimed_alive_in_sandbox(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            run = results / "test"
            run.mkdir(parents=True)
            (run / "suite.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "stages": [
                            {"name": "build", "status": "passed"},
                            {"name": "smoke_000", "status": "running"},
                        ],
                    }
                )
            )
            (run / "smoke_000.log").write_text("training setup\n")
            runtime = root / "runtime"
            runtime.mkdir()
            with sqlite3.connect(runtime / "runs.sqlite") as db:
                db.execute("CREATE TABLE runs (id TEXT, data TEXT)")
                db.execute(
                    "INSERT INTO runs VALUES (?,?)",
                    (
                        "managed",
                        json.dumps(
                            {
                                "id": "managed",
                                "request_id": "test",
                                "task_kind": "factor_sequence",
                                "phase": "running",
                                "pid": 99999999,
                                "process_created": 1,
                            }
                        ),
                    ),
                )
            before = (runtime / "runs.sqlite").read_bytes()
            report = collect(results, runtime, "test", isolated=True)
            self.assertEqual(report["current_stage"], "smoke_000")
            self.assertTrue(
                report["scheduler"][0]["process_check"].startswith("unknown")
            )
            self.assertEqual(before, (runtime / "runs.sqlite").read_bytes())
            self.assertIn("training setup", report["logs"][0]["tail"])
            self.assertIn("does not prove", report["assessment"])
            (run / "suite.json").write_text('{"status":"passed"}')
            self.assertEqual(
                collect(results, runtime, "test", isolated=True)["assessment"],
                "suite_passed",
            )

    def test_invalid_id_rejected(self):
        with self.assertRaises(ValueError):
            collect("/tmp", "/tmp", "../other")


if __name__ == "__main__":
    unittest.main()
