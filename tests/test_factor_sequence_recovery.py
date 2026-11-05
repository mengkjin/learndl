"""Recovery contracts: temporary artifacts, no database/GPU/SMTP required."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import pandas as pd
from src.data.preprocess.factor_sequence import training_mail, resume, server_suite
from src.data.preprocess.factor_sequence.bundle import write_json
from src.automation.scheduling import factor_sequence as queue
from src.automation.scheduling.factor_sequence_recovery import original_inactive

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_child_output_reaches_log_and_supervisor(self):
        from contextlib import redirect_stdout
        from io import StringIO

        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            output = StringIO()
            with redirect_stdout(output):
                server_suite.execute(
                    [sys.executable, "-c", "print('actual epoch progress')"], log, ROOT
                )
            self.assertEqual(log.read_text(), output.getvalue())
            self.assertIn("actual epoch progress", output.getvalue())
            with self.assertRaises(RuntimeError):
                server_suite.execute(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    Path(tmp) / "failed.log",
                    ROOT,
                )

    def test_request_preserves_old_schema_and_freezes_resume(self):
        with (
            TemporaryDirectory() as tmp,
            patch.object(queue, "requests_dir", return_value=Path(tmp)),
        ):
            p = queue.submit("new", resume_from="old")
            self.assertEqual(json.loads(p.read_text())["schema_version"], 2)
            store = Mock()
            store.owned.return_value = []
            self.assertEqual(queue.observe(store)["revisions"]["resume_from"], "old")
            self.assertEqual(queue.submit("new", resume_from="old"), p)
            with self.assertRaises(ValueError):
                queue.submit("new", resume_from="other")
            with self.assertRaises(ValueError):
                queue.submit("new", resume_from="../escape")

    def test_liveness_and_orphan_check(self):
        report = {
            "run_id": "old",
            "scheduler": [
                {"phase": "running", "process_check": "alive_matching_identity"}
            ],
        }
        self.assertFalse(original_inactive(report))
        report["scheduler"][0]["process_check"] = "unknown: isolated"
        with self.assertRaises(RuntimeError):
            original_inactive(report)
        report["scheduler"][0]["process_check"] = "absent"
        with patch("psutil.process_iter", return_value=[]):
            self.assertTrue(original_inactive(report))
        proc = Mock()
        proc.pid = 999999
        import os

        proc.uids.return_value.real = os.getuid()
        proc.cmdline.return_value = ["python", "--run-id", "old_f000_20170103"]
        with patch("psutil.process_iter", return_value=[proc]):
            self.assertFalse(original_inactive(report))
        with self.assertRaises(RuntimeError):
            original_inactive({"scheduler_read_error": "missing ledger"})

    def test_individual_capture_and_durable_retry(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with training_mail.capture_log(folder / "one.log"):
                print("individual epoch output")
                print("failure traceback evidence", file=sys.stderr)
            self.assertIn("individual epoch output", (folder / "one.log").read_text())
            self.assertIn(
                "failure traceback evidence", (folder / "one.log").read_text()
            )
            run = {
                "name": "one",
                "model": "factorseq_m1",
                "seed": 17,
                "status": "failed",
                "error": "boom",
            }
            with (
                patch("src.proj.MACHINE") as machine,
                patch("src.proj.util.web.task_mail.TaskResultMail") as mail,
                patch(
                    "src.proj.util.web.task_mail.retry_delivery", return_value=True
                ) as retry,
            ):
                machine.emailable = True
                mail.return_value.return_value = True
                self.assertEqual(
                    training_mail.send_training_result(folder, run)["status"], "sent"
                )
                self.assertIn(
                    folder / "one.log",
                    mail.return_value.call_args.kwargs["attachments"],
                )
                ledger = folder / "training_mail/one/delivery.json"
                ledger.write_text("{}")
                self.assertEqual(
                    training_mail.send_training_result(folder, run)["status"], "sent"
                )
                self.assertEqual(mail.return_value.call_count, 1)
                retry.assert_called_once()
                retry.side_effect = RuntimeError("SMTP ambiguous")
                self.assertEqual(
                    training_mail.send_training_result(folder, run)["status"], "failed"
                )

    def test_incomplete_fold_not_reused_complete_fold_verified(self):
        from src.res.algo.nn.model.factor_sequence import MODELS

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old"
            source.mkdir()
            target = root / "new"
            split = {"train": [1], "valid": [2], "test": [3]}
            self.assertFalse(
                resume.copy_completed_fold(
                    ROOT, source, target, split, "v", False, False
                )
            )
            training = source / "training"
            training.mkdir()
            write_json(source / "smoke.json", {"status": "passed"})
            write_json(training / "split.json", split)
            pred = pd.DataFrame(
                {
                    "date": [3, 3],
                    "secid": [10, 20],
                    "pred": [0.1, 0.2],
                    "label": [0.2, 0.3],
                }
            )
            runs = []
            for model in MODELS:
                pred.to_csv(training / f"{model}.csv", index=False)
                runs.append(
                    dict(
                        name=model,
                        model=model,
                        seed=17,
                        variant={},
                        status="passed",
                        data_version="v",
                        model_path="/unused",
                        provenance={"training_data_code": {}},
                    )
                )
            write_json(training / "runs.json", runs)
            with patch(
                "src.data.preprocess.factor_sequence.pipeline.read_predictions",
                return_value=pred,
            ):
                self.assertTrue(
                    resume.copy_completed_fold(
                        ROOT, source, target, split, "v", False, False
                    )
                )
                self.assertTrue((target / "reuse.json").exists())
                (training / f"{runs[0]['name']}.csv").write_text(
                    "date,secid,pred,label\n3,10,999,0.2\n"
                )
                with self.assertRaises(AssertionError):
                    resume.copy_completed_fold(
                        ROOT, source, root / "bad", split, "v", False, False
                    )

    def test_legacy_resume_refuses_silent_directory_migration(self):
        config = ROOT / "configs/experience/model/factor_sequence/data.json"
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "old"
            source.mkdir()
            write_json(source / "suite.json", {"status": "failed"})
            with patch.object(server_suite, "execute") as execute:
                folder, report = server_suite.run_suite(
                    ROOT, tmp, "new", config, resume_from="old"
                )
            self.assertEqual(report["status"], "failed")
            self.assertIn("Legacy per-fold", report["error"])
            self.assertEqual(execute.call_count, 1)  # Offline only, no model training.

    def test_host_checks_liveness_and_mail_failure_does_not_prevent_resume(self):
        from src.automation.scheduling import factor_sequence_recovery as recovery

        with TemporaryDirectory() as tmp:
            report = {
                "run_id": "old",
                "folds": [],
                "suite": {"status": "running"},
                "assessment": "unfinished",
            }
            module = Mock()
            module.main.return_value = 0
            with (
                patch("src.proj.PATH.result", Path(tmp)),
                patch(
                    "src.automation.scheduling.factor_sequence_progress.collect",
                    return_value=report,
                ),
                patch.object(
                    recovery, "original_inactive", return_value=False
                ) as inactive,
                patch(
                    "src.data.preprocess.factor_sequence.server_suite.send_result",
                    return_value={"status": "failed"},
                ) as summary,
                patch(
                    "src.proj.util.web.task_mail.TaskResultMail",
                    side_effect=RuntimeError("SMTP unavailable"),
                ),
            ):
                self.assertEqual(recovery.recover("new", "old", module), 75)
                summary.assert_not_called()
                module.main.assert_not_called()
                inactive.return_value = True
                self.assertEqual(recovery.recover("new", "old", module), 0)
                self.assertEqual(module.main.call_args.kwargs["resume_from"], "old")
                state = json.loads(
                    (
                        Path(tmp)
                        / "factor_sequence_diagnostics/new/host/mail_status.json"
                    ).read_text()
                )
                self.assertEqual(state["status"], "failed")

    def test_resume_rejects_revised_dataset_or_scientific_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old"
            source.mkdir()
            (root / "model.py").write_text("original")
            from src.data.preprocess.factor_sequence.bundle import file_digest

            report = {
                "dataset": str(root / "data"),
                "data_version": "v",
                "code": {"model.py": file_digest(root / "model.py")},
            }
            write_json(source / "suite.json", report)
            write_json(source / "walk_forward.json", {"folds": [{"test": [1]}]})
            with patch.object(resume, "load", return_value={"version": "v"}):
                self.assertEqual(resume.inspect_resume(root, root, "old")[1], report)
                (root / "model.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Scientific source changed"):
                    resume.inspect_resume(root, root, "old")
            with patch.object(resume, "load", return_value={"version": "other"}):
                with self.assertRaisesRegex(ValueError, "dataset version changed"):
                    resume.inspect_resume(root, root, "old")

    def test_suite_mail_retries_existing_delivery_without_repackaging(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "mail").mkdir()
            write_json(folder / "suite.json", {"status": "passed"})
            (folder / "mail/delivery.json").write_text("{}")
            with (
                patch("src.proj.MACHINE") as machine,
                patch(
                    "src.proj.util.web.task_mail.retry_delivery", return_value=True
                ) as retry,
                patch.object(server_suite, "report_package") as package,
            ):
                machine.emailable = True
                self.assertEqual(server_suite.send_result(folder)["status"], "sent")
                retry.assert_called_once()
                package.assert_not_called()


if __name__ == "__main__":
    unittest.main()
