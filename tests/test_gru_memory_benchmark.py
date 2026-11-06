"""Verify benchmark semantics and reporting without sending mail or using CUDA."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/5_test/benchmark_gru_memory.py"
spec = importlib.util.spec_from_file_location("gru_memory_benchmark", SCRIPT)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def tiny_config():
    args = bench.parser().parse_args(["--stocks", "3", "--hidden", "4", "--days", "3",
                                     "--bars", "2", "--features", "2", "--day-hidden", "4",
                                     "--chunks", "4", "--dropout", "0", "--steps", "2"])
    return next(bench.make_cases(args))


class BenchmarkTests(unittest.TestCase):
    def test_all_modes_preserve_outputs_gradients_and_bn_updates(self):
        config = tiny_config()
        torch.manual_seed(31)
        x = torch.randn(3, 3, 2, 2)
        target = torch.randn(3, 1)
        reference = None
        for mode in bench.MODES:
            torch.manual_seed(17)
            net = bench.SyntheticGRU(config | {"mode": mode, "chunk_size": 4})
            out = net(x)
            (out - target).square().mean().backward()
            result = (out.detach(), [p.grad for p in net.parameters()], list(net.buffers()))
            self.assertTrue(all(grad is not None for grad in result[1]))
            if reference is None:
                reference = result
            else:
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)

    def test_checkpoint_preserves_dropout_with_fixed_chunk_layout(self):
        config = tiny_config() | {"dropout": .1, "chunk_size": 4, "day_checkpoint": True}
        x, target = torch.randn(3, 3, 2, 2), torch.randn(3, 1)
        results = []
        for mode in ("chunk", "chunk_checkpoint", "chunk_outer_checkpoint"):
            torch.manual_seed(20)
            net = bench.SyntheticGRU(config | {"mode": mode})
            out = net(x)
            (out - target).square().mean().backward()
            results.append((out, [p.grad for p in net.parameters()], list(net.buffers())))
        for result in results[1:]:
            torch.testing.assert_close(result, results[0])

    def test_cpu_smoke_is_not_reported_as_gpu_memory(self):
        result = bench.run_case(tiny_config(), "cpu")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["steps"]), 2)
        self.assertIsNone(bench.summarize(result)["peak_allocated_gib"])
        self.assertGreater(bench.summarize(result)["warm_step_seconds"], 0)

    def test_oom_is_classified_at_current_phase(self):
        with patch.object(bench.SyntheticGRU, "forward", side_effect=torch.cuda.OutOfMemoryError("test")):
            result = bench.run_case(tiny_config(), "cpu")
        self.assertEqual(result["status"], "oom")
        self.assertEqual(result["stage"], "forward")

    def test_each_case_continues_after_oom_and_timeout_and_mails_reports(self):
        count = 0

        def fake_run(command, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise subprocess.TimeoutExpired(command, 1)
            config_path = Path(command[command.index("--worker-config") + 1])
            config = json.loads(config_path.read_text())
            bench.save_json(config_path.with_suffix(".result.json"),
                            dict(config=config, status="oom", stage="backward", error="synthetic OOM"))
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp, patch.object(bench.subprocess, "run", side_effect=fake_run), \
                patch.object(bench, "send_results") as mail:
            directory = Path(tmp) / "report"
            code = bench.main(["--output-dir", str(directory), "--hidden", "4", "--modes",
                               "full", "full_checkpoint", "chunk", "--chunks", "4"])
            self.assertEqual(code, 1)
            self.assertEqual(count, 3)
            report = json.loads((directory / "results.json").read_text())
            self.assertEqual([case["status"] for case in report["cases"]], ["oom", "timeout", "oom"])
            self.assertTrue((directory / "results.csv").exists())
            self.assertIn("lower bounds", (directory / "summary.txt").read_text())
            mail.assert_called_once_with(directory.resolve(), None)

    def test_email_failure_preserves_results_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench, "make_cases", return_value=[]), \
                patch.object(bench, "send_results", side_effect=RuntimeError("SMTP unavailable")):
            directory = Path(tmp) / "report"
            self.assertEqual(bench.main(["--output-dir", str(directory)]), 1)
            self.assertTrue((directory / "results.json").exists())

    def test_mail_uses_existing_delivery_receipts_and_attaches_reports(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bench.write_reports(directory, dict(host="test", status="completed", cases=[]))
            sender = Mock(return_value=True)
            with patch("src.proj.env.MACHINE.emailable", True), \
                    patch("src.proj.util.web.task_mail.TaskResultMail", return_value=sender) as factory:
                bench.send_results(directory, "test@example.com")
            factory.assert_called_once_with(directory / "mail")
            self.assertEqual(sender.call_args.args[2], "test@example.com")
            self.assertEqual({path.name for path in sender.call_args.kwargs["attachments"]},
                             {"results.json", "summary.txt"})
            with patch("src.proj.env.MACHINE.emailable", False):
                with self.assertRaisesRegex(RuntimeError, "NOT emailed"):
                    bench.send_results(directory, None)

    def test_default_matrix_has_22_cases(self):
        self.assertEqual(len(list(bench.make_cases(bench.parser().parse_args([])))), 22)


if __name__ == "__main__":
    unittest.main()
