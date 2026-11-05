"""Differential checks for common generators and default-parameter fallback."""

from contextlib import ExitStack
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path

import numpy as np
from test_t50_array import Market, assert_groups
from src.res.factor.fmp.builder import PortfolioGroupBuilder
from src.res.factor.fmp.generator_array import run_generator
from src.res.factor.fmp.prepared_market import prepare_market, PreparedMarket
from src.res.factor.util.classes.alpha_model import AlphaComponent


class GeneratorArrayTest(TestCase):
    def test_four_generators_and_parameter_fallback(self):
        market = Market(days=7)
        with ExitStack() as stack:
            market.install(stack)
            seed = market.group(models=2)
            # Distinct external signal, supplied through the real component API.
            original = AlphaComponent.get_alpha_model
            stack.enter_context(
                patch.object(
                    AlphaComponent,
                    "get_alpha_model",
                    autospec=True,
                    side_effect=lambda comp, dates: (
                        seed.alpha_models[1]
                        if isinstance(comp.input, str)
                        else original(comp, dates)
                    ),
                )
            )
            for category in ("top", "screen", "revscreen", "reinforce"):

                def group(**kw):
                    return PortfolioGroupBuilder(
                        category,
                        [seed.alpha_models[0]],
                        seed.benchmarks,
                        analytic=False,
                        attribution=False,
                        trade_engine="yale",
                        start=int(market.dates[0]),
                        end=int(market.dates[-1]),
                        vb_level="never",
                        **kw,
                    )

                old, new = group(), group()
                a, b = run_generator(old, "legacy"), run_generator(new, "array")
                assert_groups(self, old, new, a, b)
                changed = group(screen_ratio=0.6)
                with patch.object(changed, "build", wraps=changed.build) as build:
                    run_generator(changed, "auto")
                    build.assert_called_once()

    def test_prepared_quotes_equal_and_revision_rejected(self):
        market = Market(days=4)
        with ExitStack() as stack, TemporaryDirectory() as tmp:
            market.install(stack)
            revision = {"value": 1}
            stack.enter_context(
                patch(
                    "src.res.factor.fmp.prepared_market.fingerprints",
                    side_effect=lambda dates: dict(revision),
                )
            )
            report = prepare_market(tmp, market.dates)
            self.assertGreater(report["bytes"], 0)
            prepared = PreparedMarket(tmp)
            for date in market.dates:
                quote = prepared.quote(int(date), int(date) + 1, "open")
                assert quote is not None
                ids, values = quote
                q = market.quote(date, date + 1, price1="open").sort_index()
                np.testing.assert_array_equal(ids, q.index)
                np.testing.assert_allclose(values, q.ret, equal_nan=True)
            self.assertIsNone(
                prepared.quote(int(market.dates[0]), int(market.dates[-1]), "close")
            )
            old, new = market.group(models=1), market.group(models=1)
            a = run_generator(old, "legacy")
            with patch.object(
                PreparedMarket, "quote", autospec=True, side_effect=PreparedMarket.quote
            ) as quote:
                b = run_generator(new, "array", prepared_root=tmp)
                self.assertTrue(
                    any(
                        call.args[2] == call.args[1] + 1
                        for call in quote.call_args_list
                    )
                )
            assert_groups(self, old, new, a, b)
            revision["value"] = 2
            with self.assertRaises(ValueError):
                PreparedMarket(tmp)
            fallback = market.group(models=1)
            with patch.object(fallback, "build", wraps=fallback.build) as build:
                run_generator(fallback, "auto", prepared_root=tmp)
                build.assert_called_once()

    def test_resume_and_sparse_periods(self):
        from src.proj import Save

        market = Market(days=10)
        with ExitStack() as stack, TemporaryDirectory() as tmp:
            market.install(stack, resume=True)
            stack.enter_context(
                patch(
                    "src.res.factor.fmp.prepared_market.fingerprints", return_value={}
                )
            )
            prepared = Path(tmp) / "market"
            prepare_market(prepared, market.dates)
            for dates in (market.dates[:6], market.dates[4:], market.dates[4:]):
                groups = []
                frames = []
                for mode in ("legacy", "array"):
                    group = market.group(
                        dates, models=1, resume=True, resume_path=Path(tmp) / mode
                    )
                    frames.append(
                        run_generator(
                            group,
                            mode,
                            prepared_root=prepared if mode == "array" else None,
                            block_days=3,
                        )
                    )
                    Save.async_wait_all(caller_name="generator_resume_test")
                    groups.append(group)
                assert_groups(self, *groups, *frames)
            old, new = (
                market.group(market.dates[::3], models=1),
                market.group(market.dates[::3], models=1),
            )
            a = run_generator(old, "legacy")
            market.quote_calls.clear()
            b = run_generator(new, "array", prepared_root=prepared)
            self.assertTrue(
                any(end > begin + 1 for begin, end, _ in market.quote_calls)
            )
            assert_groups(self, old, new, a, b)

    def test_fmp_preserves_changed_nbest(self):
        import importlib
        from types import SimpleNamespace

        for kind in ("t50", "screen", "revscreen", "reinforce"):
            module = importlib.import_module(f"src.res.factor.analytic.fmp_{kind}")
            cls = next(
                v
                for k, v in vars(module).items()
                if isinstance(v, type) and k.endswith("FMPTest")
            )
            obj = SimpleNamespace(kwargs={"n_best": 20})
            cls.update_kwargs(obj)
            self.assertEqual(obj.kwargs["n_best"], 20)

    def test_changed_parameter_never_opens_prepared_pack(self):
        market = Market(days=3)
        with ExitStack() as stack:
            market.install(stack)
            group = market.group(turn_control=0.2, models=1)
            with patch(
                "src.res.factor.fmp.prepared_market.PreparedMarket",
                side_effect=AssertionError("must not load mismatched pack"),
            ):
                with patch.object(group, "build", wraps=group.build) as build:
                    run_generator(group, "auto", prepared_root="unused")
                    build.assert_called_once()
