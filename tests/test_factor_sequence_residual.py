import unittest
import torch
from src.res.algo.nn.model.factor_sequence.residual import FactorResidual


class ResidualTests(unittest.TestCase):
    def test_modes_linear_initialization_gradients_and_roundtrip(self):
        for mode in ("snapshot", "temporal", "contextual"):
            dims = [8, 12, 6] if mode == "contextual" else [8]
            m = FactorResidual(dims, mode=mode, dropout=0)
            x = [torch.randn(7, 1 if mode == "snapshot" else 4, 8)]
            if mode == "contextual":
                x += [torch.randn(1, 8, 12), torch.randn(1, 8, 6)]
            y, out = m(x)
            torch.testing.assert_close(y, m.linear(x[0][:, -1, :2]))
            self.assertEqual(out["hidden"].shape, (7, 32))
            optim = torch.optim.Adam(m.parameters())
            for _ in range(3):
                optim.zero_grad()
                y, _ = m(x)
                y.square().mean().backward()
                optim.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
            self.assertGreater(m.head[0].weight.grad.abs().sum().item(), 0)
            clone = FactorResidual(dims, mode=mode, dropout=0)
            clone.load_state_dict(m.state_dict())
            torch.testing.assert_close(m(x)[0], clone(x)[0])

    def test_film_starts_as_temporal_identity(self):
        b = FactorResidual([8], mode="temporal", dropout=0)
        c = FactorResidual([8, 12, 6], mode="contextual", dropout=0)
        c.load_state_dict(b.state_dict(), strict=False)
        torch.nn.init.normal_(b.output.weight)
        c.output.load_state_dict(b.output.state_dict())
        x = torch.randn(5, 4, 8)
        torch.testing.assert_close(
            b(x)[0], c([x, torch.randn(1, 8, 12), torch.randn(1, 8, 6)])[0]
        )


if __name__ == "__main__":
    unittest.main()


class CompactScheduleTests(unittest.TestCase):
    def test_expanding_schedule_and_streams(self):
        import json
        import tempfile
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        from src.data.preprocess.factor_sequence.worklist import resolve_schedule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "factor_sequence" / "old"
            suite.mkdir(parents=True)
            dataset = root / "dataset"
            dataset.mkdir()
            dates = np.arange(20000000, 20002000)
            np.savez(dataset / "factorseq.npz", dates=dates)
            (dataset / "audit.json").write_text(
                json.dumps({"status": "passed", "version": "v"})
            )
            (suite / "suite.json").write_text(
                json.dumps({"dataset": str(dataset), "data_version": "v"})
            )
            folds = [
                dict(
                    train=[1],
                    valid=[2],
                    test=dates[a : a + 63].tolist(),
                    model_date=int(dates[a] - 1),
                )
                for a in (1450, 1800)
            ]
            (suite / "walk_forward.json").write_text(json.dumps({"folds": folds}))
            with (
                patch("src.proj.PATH.result", root),
                patch("src.proj.CALENDAR.td", side_effect=lambda d, n: int(d) + n),
                patch(
                    "src.data.preprocess.factor_sequence.bundle.load",
                    return_value={"version": "v"},
                ),
            ):
                for mode in ("snapshot", "temporal", "contextual"):
                    c = resolve_schedule("old", "factorseq_residual", "test", mode)
                    self.assertEqual(c["train.max_epoch"], 30)
                    self.assertIn("DetailedAlphaAnalysis", c["train.callbacks"])
                    f = c["train.factor_sequence.folds"]
                    self.assertFalse(f[0]["full_window"])
                    self.assertTrue(f[1]["full_window"])
                    self.assertLess(max(f[0]["train"]) + 11, min(f[0]["valid"]))
                    self.assertEqual(
                        c["input.data.types"],
                        "factorseq+factorperf+factormarket"
                        if mode == "contextual"
                        else "factorseq",
                    )


class CompactLoaderTests(unittest.TestCase):
    def test_compact_selection_and_alignment_hole(self):
        import tempfile
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        from src.data.preprocess.factor_sequence.processors import load_block
        from src.data.preprocess.factor_sequence.schema import STOCK_CHANNELS

        names = [f"{c}__a" for c in STOCK_CHANNELS]
        manifest = {
            "config": {"factors": ["a"]},
            "blocks": {"factorseq": {"features": names}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(
                Path(tmp) / "factorseq.npz",
                values=np.arange(10, dtype="float32").reshape(1, 1, 1, 10),
                features=names,
                dates=[20170103],
                secids=[1],
            )
            with patch(
                "src.data.preprocess.factor_sequence.processors.load_manifest",
                return_value=manifest,
            ):
                block = load_block(
                    tmp, "factorseq", dates=[20170103, 20170104], view="compact"
                )
            self.assertEqual(
                list(block.feature), ["value__a", "valid__a", "gap__a", "never__a"]
            )
            torch.testing.assert_close(
                block.values[0, 0, 0], torch.tensor([0.0, 1.0, 3.0, 4.0])
            )
            torch.testing.assert_close(
                block.values[0, 1, 0], torch.tensor([0.0, 0.0, 0.0, 1.0])
            )
