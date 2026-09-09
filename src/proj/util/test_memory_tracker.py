from __future__ import annotations

import json
import os
import tempfile
import unittest

from pathlib import Path

from src.proj.util.memory_tracker import (
    ProcessMemoryTracker,
    _add_cgroup_metrics,
    _summarize_report,
    collect_memory_sample,
)


class TestProcessMemoryTracker(unittest.TestCase):
    def test_tracker_flushes_phase_markers_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with ProcessMemoryTracker(
                "unit test", interval=60, stdout_interval=0, output_dir=output_dir,
                launch_watcher=False, register_output=False, enabled=True,
            ) as tracker:
                tracker.mark("loader_static_clone_end", shape=[3, 4, 5])
                report_path = tracker.path
                done_path = tracker.done_path

            self.assertIsNotNone(report_path)
            self.assertIsNotNone(done_path)
            assert report_path is not None
            assert done_path is not None
            records = [json.loads(line) for line in report_path.read_text().splitlines()]
            self.assertEqual(records[0]["event"], "start")
            self.assertTrue(any(record.get("phase") == "loader_static_clone_end" for record in records))
            self.assertEqual(records[-1]["event"], "stop")
            self.assertTrue(done_path.exists())

    def test_tracker_is_a_noop_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with ProcessMemoryTracker(
                "disabled", output_dir=Path(tmp), launch_watcher=False,
                register_output=False, enabled=False,
            ) as tracker:
                tracker.mark("ignored")

            self.assertIsNone(tracker.path)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_report_summary_recovers_peak_and_last_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "memory.jsonl"
            rows = [
                {
                    "event": "mark", "timestamp": "first", "phase": "clone",
                    "process_vmrss_bytes": 10, "system_memavailable_bytes": 100,
                },
                {
                    "event": "sample", "timestamp": "last", "phase": "materialize",
                    "process_vmrss_bytes": 20, "system_memavailable_bytes": 50,
                    "process_mems_allowed": "0-3",
                },
            ]
            report_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            summary = _summarize_report(report_path)

            self.assertIn("Last phase: materialize", summary)
            self.assertIn("Process NUMA nodes allowed: 0-3", summary)
            self.assertIn("Minimum system available memory", summary)

    def test_sample_is_json_serializable_on_this_platform(self) -> None:
        sample = collect_memory_sample(os.getpid())
        self.assertEqual(sample["pid"], os.getpid())
        self.assertIsInstance(json.dumps(sample), str)

    def test_cgroup_v2_values_and_pressure_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cgroup_dir = Path(tmp)
            (cgroup_dir / "memory.current").write_text("1234\n", encoding="utf-8")
            (cgroup_dir / "memory.max").write_text("max\n", encoding="utf-8")
            (cgroup_dir / "memory.events").write_text("oom 2\noom_kill 1\n", encoding="utf-8")
            (cgroup_dir / "memory.pressure").write_text(
                "some avg10=51.25 avg60=30.00 avg300=10.00 total=99\n",
                encoding="utf-8",
            )
            sample: dict[str, object] = {}
            _add_cgroup_metrics(sample, cgroup_dir, "scope")

            self.assertEqual(sample["scope_memory_current_bytes"], 1234)
            self.assertEqual(sample["scope_memory_max_bytes"], "max")
            self.assertEqual(sample["scope_event_oom_kill"], 1)
            self.assertEqual(sample["scope_psi_some_avg10"], 51.25)


if __name__ == "__main__":
    unittest.main()
