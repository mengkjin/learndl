import tempfile
from typing import Any, cast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.data.preprocess.factor_sequence.context_audit import (
    dictionary,
    PERF_NAMES,
    MARKET_NAMES,
    EVENTS,
    RISK_NAMES,
    CoverageState,
    candidate_market,
    context_quality,
)
from src.data.preprocess.factor_sequence.risk_audit import minute_reference
from src.data.preprocess.factor_sequence.context_diagnostics import (
    film_probe,
    paired_days,
)
from src.data.preprocess.factor_sequence.audit_runner import read_factor
from src.res.algo.nn.model.factor_sequence.residual import FactorResidual


class ContextAuditTests(unittest.TestCase):
    def market(self, n=180):
        x = pd.DataFrame(index=np.arange(n))
        for e in EVENTS:
            x[e] = 0.0
            x.loc[20, e] = 1.0
        x["market_return"] = np.resize([-0.01, 0, 0.01, np.nan], n)
        for k in RISK_NAMES:
            x[k] = np.arange(n, dtype=float)
        return x

    def test_contract_and_prefix(self):
        self.assertEqual(len(PERF_NAMES), 66)
        self.assertEqual(len(MARKET_NAMES), 15)
        self.assertFalse(dictionary()["training_enabled"])
        raw = self.market()
        a = candidate_market(raw)
        pd.testing.assert_frame_equal(a.iloc[:140], candidate_market(raw.iloc[:140]))
        self.assertTrue(a.loc[:19, EVENTS[0] + "__age"].isna().all())
        self.assertTrue(a.loc[:59, RISK_NAMES[0]].isna().all())
        self.assertTrue(pd.notna(a.loc[60, RISK_NAMES[0]]))
        self.assertEqual(a.loc[1, "market_up"], 0)
        self.assertTrue(pd.isna(a.loc[3, "market_up"]))
        raw.loc[25, EVENTS[0]] = np.nan
        b = candidate_market(raw)
        self.assertTrue(pd.isna(b.loc[26, EVENTS[0] + "__age"]))

    def test_window_and_universe_boundaries(self):
        state = CoverageState(("a", "b"), window=3)
        full = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[1, 2])
        r, _, _ = state.update(1, full, [1])
        self.assertTrue(np.isnan(r["complete_60_rate"]))
        state.update(2, full, [1, 2])
        r, _, _ = state.update(3, full, [1, 2])
        self.assertEqual(r["complete_60_rate"], 0.5)  # newly eligible not full window
        r, _, _ = state.update(4, full, [2])
        self.assertEqual(
            r["complete_60_rate"], 1.0
        )  # delisted stock excluded denominator
        with self.assertRaises(ValueError):
            state.update(4, full, [2])

    def test_missing_and_leave_one_out(self):
        state = CoverageState(("a", "b"))
        f = pd.DataFrame({"a": [0, np.nan, 2], "b": [1, 2, np.inf]}, index=[1, 2, 3])
        r, s, _ = state.update(1, f, [1, 2, 3, 4])
        self.assertEqual(r["complete_rate"], 0.25)
        self.assertEqual(s[0]["leave_one_out_gain"], 0.25)
        self.assertEqual(s[0]["coverage"], 0.5)

    def test_chunk_state_equivalent(self):
        f = pd.DataFrame({"a": [1.0]}, index=[1])
        a, b = CoverageState(("a",)), CoverageState(("a",))
        allrows = [a.update(d, f, [1])[0] for d in range(70)]
        chunkrows = []
        for chunk in (range(25), range(25, 70)):
            chunkrows.extend(b.update(d, f, [1])[0] for d in chunk)
        pd.testing.assert_frame_equal(pd.DataFrame(allrows), pd.DataFrame(chunkrows))
        q = context_quality(pd.DataFrame({"x": np.ones(130)}))
        self.assertTrue(np.isnan(q.complete_120.iloc[118]))
        self.assertEqual(q.complete_120.iloc[119], 1)

    def test_source_states_and_corrupt_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factor"
            db = SimpleNamespace(path=lambda *a: path)
            calc = SimpleNamespace(
                db_src="s", db_key="k", Loads=lambda *a, **k: pd.DataFrame({"a": [1]})
            )
            _, r = read_factor(calc, "a", 1, [1], db, 2)
            self.assertEqual(r["status"], "not_started")
            _, r = read_factor(calc, "a", 3, [1], db, 2)
            self.assertEqual(r["status"], "file_missing")
            path.write_text("x")
            _, r = read_factor(calc, "a", 3, [1], db, 2)
            self.assertEqual(r["status"], "corrupt")

    def test_grouped_minute_returns_not_security_boundary(self):
        x = pd.DataFrame(
            {
                "secid": [2, 1, 2, 1],
                "minute": [1, 0, 0, 1],
                "close": [220.0, 10.0, 200.0, 11.0],
            }
        )
        r = minute_reference(x, pd.Series([1.0, 3.0], index=[1, 2])).set_index("minute")
        self.assertAlmostEqual(float(cast(Any, r.loc[1, "reference"])), 0.1)
        self.assertEqual(r.loc[0, "valid_weight_coverage"], 0)
        self.assertTrue(pd.isna(r.loc[0, "reference"]))
        x.loc[0, "close"] = np.nan
        r = minute_reference(x, pd.Series([1.0, 3.0], index=[1, 2])).set_index("minute")
        self.assertEqual(r.loc[1, "valid_weight_coverage"], 0.25)
        self.assertAlmostEqual(
            float(cast(Any, r.loc[1, "production_denominator_reference"])), 0.025
        )

    def test_probe_restores_model_and_gradients(self):
        m = FactorResidual([8, 4, 6], mode="contextual", dropout=0)
        torch.nn.init.normal_(m.output.weight)
        x = [torch.randn(5, 4, 8), torch.randn(1, 8, 4), torch.randn(1, 8, 6)]
        before = {n: p.clone() for n, p in m.state_dict().items()}
        report = film_probe(m, x, torch.randn(5), [0, 1])
        self.assertEqual(report["disable_film_rms"], 0.0)
        self.assertTrue(m.training)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        for n, p in m.state_dict().items():
            torch.testing.assert_close(p, before[n])

    def test_pairing_rejects_sample_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir()
            b.mkdir()
            f = pd.DataFrame(
                dict(
                    secid=[1, 2, 3],
                    label=[1.0, 2.0, 3.0],
                    pred=[1.0, 2.0, 3.0],
                    linear=[0.0, 1.0, 2.0],
                    residual=[1.0, 1.0, 1.0],
                    coverage=[1.0, 1.0, 1.0],
                )
            )
            f.to_feather(a / "20200101.feather")
            f.to_feather(b / "20200101.feather")
            self.assertEqual(paired_days(a, b).iloc[0].delta_ic, 0)
            f.loc[0, "secid"] = 4
            f.to_feather(b / "20200101.feather")
            with self.assertRaises(ValueError):
                paired_days(a, b)

    def test_empty_delivery_is_not_success(self):
        from src.automation.artifact_exchange.research import finalize_task_artifacts

        owner = SimpleNamespace(
            _artifacts=[], _artifact_results=[], _artifact_delivery=None
        )
        with patch("src.automation.artifact_exchange.research.delivery_log") as log:
            finalize_task_artifacts(owner)
        self.assertEqual(owner._artifact_delivery_summary["unregistered"], 1)
        self.assertEqual(owner._artifact_delivery_summary["published"], 0)
        self.assertIn("不能视为投递成功", log.call_args.args[0])


class AuditDeliveryTests(unittest.TestCase):
    def test_mail_uses_complete_csv_zip_and_mock_sender(self):
        import json
        from zipfile import ZipFile
        from src.data.preprocess.factor_sequence.audit_runner import deliver

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "summary.json").write_text(
                json.dumps({"status": "incomplete", "run_id": "test"})
            )
            (folder / "feature_dictionary.json").write_text("{}")
            (folder / "coverage_daily.csv").write_text("date,complete\n20110104,0\n")
            with (
                patch(
                    "src.data.preprocess.factor_sequence.context_diagnostics.recover_reports",
                    return_value=[],
                ),
                patch(
                    "src.automation.artifact_exchange.research.publish_safe",
                    return_value=[{"status": "prepared"}],
                ),
                patch(
                    "src.automation.artifact_exchange.research.defer_mail",
                    return_value=None,
                ),
                patch("src.proj.MACHINE.emailable", True),
                patch("src.proj.util.web.task_mail.TaskResultMail") as sender,
            ):
                sender.return_value.return_value = True
                result = deliver(folder)
            self.assertEqual(result["status"], "sent")
            self.assertIn(
                folder / "daily_results.zip",
                sender.return_value.call_args.kwargs["attachments"],
            )
            with ZipFile(folder / "daily_results.zip") as z:
                self.assertIn("coverage_daily.csv", z.namelist())


if __name__ == "__main__":
    unittest.main()
