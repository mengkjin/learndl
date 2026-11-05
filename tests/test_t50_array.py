"""Differential tests against the actual Pandas top/yale implementation."""

from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, main
from unittest.mock import patch, PropertyMock, Mock

import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.proj import CALENDAR, Const, Save
from src.res.factor.fmp.builder import PortfolioGroupBuilder
from src.res.factor.fmp.generator.basic import BasicCreatorConfig
from src.res.factor.fmp.generator_array import (
    Position,
    Quotes,
    Inputs,
    UnsupportedGenerator,
    account_step,
    descending_order,
    evolve,
    from_port,
    prepare_selection,
    run_generator,
    select,
    turnover,
    weighted_return,
)
from src.res.factor.util import AlphaModel, Amodel, Portfolio, Port, RISK_MODEL


class Market:
    """Nontrivial returns, changing industry/pool, missing quote rows and NaNs."""

    def __init__(self, days=12, stocks=180):
        self.dates = np.arange(20230102, 20230102 + days)
        self.ids = np.arange(1, stocks + 1)
        self.quote_calls = []

    def quote(self, begin, end, price0="close", price1="close", secid=None):
        self.quote_calls.append((int(begin), int(end), price1))
        ids = self.ids[(self.ids + begin) % 97 != 0]
        values = np.sin(ids * 0.7 + begin) * 0.01 * (end - begin)
        if price1 != "close":
            values += 0.003 if price1 == "open" else -0.002
        values = values.astype(float)
        values[(ids + end) % 89 == 0] = np.nan
        return pd.DataFrame({"secid": ids, "ret": values}).set_index("secid")

    def industry(self, date):
        values = np.array([f"i{x % 16}" for x in self.ids], dtype=object)
        values[(self.ids + date) % 31 == 0] = None
        return pd.DataFrame({"indus": values}, index=pd.Index(self.ids, name="secid"))

    def install(self, stack, resume=False):
        stack.enter_context(
            patch.object(DATAVENDOR, "get_quote_ret", side_effect=self.quote)
        )
        stack.enter_context(
            patch.object(DATAVENDOR, "td", side_effect=lambda d, n=0: int(d) + n)
        )
        stack.enter_context(
            patch.object(
                DATAVENDOR, "td_within", side_effect=lambda a, b: np.arange(a, b + 1)
            )
        )
        stack.enter_context(
            patch.object(DATAVENDOR.INFO, "get_indus", side_effect=self.industry)
        )
        stack.enter_context(
            patch.object(
                CALENDAR, "offset", side_effect=lambda d, n, *a: np.asarray(d) + n
            )
        )
        stack.enter_context(
            patch.object(CALENDAR, "updated", return_value=int(self.dates[-1]) + 10)
        )
        stack.enter_context(patch.object(RISK_MODEL, "load_models"))
        stack.enter_context(
            patch.object(
                type(Const.Model),
                "resume_fmp_account",
                new_callable=PropertyMock,
                return_value=resume,
            )
        )

    def group(self, dates=None, models=2, **kwargs):
        dates = self.dates if dates is None else dates
        alphas = []
        for j in range(models):
            days = []
            for i, date in enumerate(dates):
                # Each submodel has a different ordering; several ties cross
                # selection and industry thresholds, plus missing alpha.
                values = np.round(np.sin(self.ids * 0.27 + date + j), 1)
                values[(self.ids + date) % 71 == 0] = np.nan
                if i % 7 == 3 and j == 1:
                    continue
                days.append(Amodel(int(date), values, self.ids.copy(), f"a{j}"))
            alphas.append(AlphaModel(f"a{j}", days))
        universe = Portfolio("univ")
        for date in self.dates:
            ids = self.ids[(self.ids + date) % 13 != 0]
            universe.append(
                Port(
                    pd.DataFrame(
                        {"secid": ids, "weight": np.full(len(ids), 1 / len(ids))}
                    ),
                    int(date),
                    "univ",
                )
            )
        return PortfolioGroupBuilder(
            "top",
            alphas,
            [universe],
            analytic=False,
            attribution=False,
            trade_engine="yale",
            start=int(self.dates[0]),
            end=int(dates[-1]),
            n_best=50,
            vb_level="never",
            **kwargs,
        )


def assert_groups(test, legacy, fast, a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    for left, right in zip(legacy.builders, fast.builders, strict=True):
        test.assertEqual(list(left.portfolio.ports), list(right.portfolio.ports))
        for date, port in left.portfolio.ports.items():
            other = right.portfolio.ports[date]
            np.testing.assert_array_equal(port.secid, other.secid)
            np.testing.assert_allclose(
                port.weight, other.weight, rtol=1e-10, atol=1e-12
            )
            np.testing.assert_allclose(port.value, other.value, rtol=1e-10, atol=1e-12)


class T50ArraysTest(TestCase):
    def test_sort_and_selection_ties_nan_empty_and_parameter_changes(self):
        rng = np.random.default_rng(28)
        market = Market()
        with ExitStack() as stack:
            market.install(stack)
            for n in (0, 1, 20, 180):
                for repeat in range(6):
                    ids = market.ids[:n]
                    alpha = rng.integers(-3, 4, n).astype(float)
                    if n:
                        alpha[::11] = np.nan
                    np.testing.assert_array_equal(
                        descending_order(alpha),
                        pd.Series(alpha).sort_values(ascending=False).index.to_numpy(),
                    )
                    conf = BasicCreatorConfig(
                        n_best=50,
                        turn_control=repeat / 6,
                        buffer_zone=0.8,
                        no_zone=0.5,
                        indus_control=0.1 + repeat * 0.05,
                    )
                    amodel = Amodel(20230102, alpha, ids)
                    old = Port(
                        pd.DataFrame(
                            {"secid": ids[::2], "weight": np.full(len(ids[::2]), 0.02)}
                        ),
                        20230102,
                    )
                    prepared_alpha = (
                        conf.get_sorting_alpha(20230102, amodel).align(ids).alpha
                        if n
                        else alpha
                    )
                    labels = (
                        market.industry(20230102)
                        .reindex(ids)["indus"]
                        .fillna("unknown")
                    )
                    day = prepare_selection(
                        ids, prepared_alpha, pd.factorize(labels)[0]
                    )
                    selected = select(day, from_port(old), conf, 20230102)
                    if n:  # empty Amodel is separately exercised with an empty eligible pool
                        expected = conf.generate_top_stock_port(20230102, amodel, old)
                        np.testing.assert_array_equal(selected.ids, expected.secid)
                    else:
                        self.assertEqual(len(selected.ids), 0)

    def test_sparse_returns_and_evolution_match_port(self):
        market = Market()
        with ExitStack() as stack:
            market.install(stack)
            port = Port(
                pd.DataFrame(
                    {
                        "secid": market.ids,
                        "weight": np.full(len(market.ids), 1 / len(market.ids)),
                    }
                ),
                20230102,
            )
            inp = Inputs()
            q = inp.quote(20230102, 20230105)
            np.testing.assert_allclose(
                weighted_return(from_port(port), q),
                port.fut_ret(20230105),
                rtol=1e-10,
                atol=1e-12,
            )
            expected = port.evolve_to_date(20230105)
            actual = evolve(from_port(port), 20230105, q)
            np.testing.assert_array_equal(actual.ids, expected.secid)
            np.testing.assert_allclose(
                actual.weights, expected.weight, rtol=1e-10, atol=1e-12
            )
            self.assertEqual(actual.value, expected.value)
            self.assertEqual(
                turnover(actual, from_port(port), Const.Factor.ROUNDING.turnover),
                expected.turnover(port),
            )

    def test_complete_pipeline_multiple_models_blocks_and_no_cache_leakage(self):
        market = Market()
        with ExitStack() as stack:
            market.install(stack)
            legacy = market.group()
            a = run_generator(legacy, "legacy")
            for size in (1, 5, 128):
                fast = market.group()
                b = run_generator(fast, "array", block_days=size)
                assert_groups(self, legacy, fast, a, b)

    def test_resume_trailing_and_no_new_account_dates(self):
        market = Market(days=10)
        with TemporaryDirectory() as root, ExitStack() as stack:
            market.install(stack, resume=True)
            paths = [Path(root) / name for name in ("legacy", "array")]
            for dates in (market.dates[:6], market.dates[4:], market.dates[4:]):
                groups, frames = [], []
                for path, mode in zip(paths, ("legacy", "array")):
                    group = market.group(dates, resume=True, resume_path=path)
                    frames.append(run_generator(group, mode, block_days=3))
                    Save.async_wait_all(caller_name="test_t50_array")
                    groups.append(group)
                assert_groups(self, *groups, *frames)

    def test_auto_falls_back_only_for_unsupported_contract(self):
        market = Market(days=3)
        with ExitStack() as stack:
            market.install(stack)
            group = market.group()
            group.acc_kwargs["daily"] = True
            with self.assertRaises(UnsupportedGenerator):
                run_generator(group, "array")
            with (
                patch.object(group, "build", return_value=group) as build,
                patch.object(group, "total_account", return_value=pd.DataFrame()),
            ):
                run_generator(group, "auto")
                build.assert_called_once()
            group = market.group()
            with patch(
                "src.res.factor.fmp.generator_array.generate_arrays",
                side_effect=RuntimeError("bad code"),
            ):
                with self.assertRaisesRegex(RuntimeError, "bad code"):
                    run_generator(group, "auto")

    def test_real_128_day_boundary_and_sparse_prediction_schedule(self):
        market = Market(days=132, stocks=70)
        with ExitStack() as stack:
            market.install(stack)
            legacy, fast = market.group(models=1), market.group(models=1)
            a, b = run_generator(legacy, "legacy"), run_generator(fast, "array")
            assert_groups(self, legacy, fast, a, b)
            # A single model with non-daily predictions has multi-day account
            # periods, rather than another model filling the missing dates.
            dates = market.dates[::9]
            legacy, fast = market.group(dates, models=1), market.group(dates, models=1)
            a, b = run_generator(legacy, "legacy"), run_generator(fast, "array")
            assert_groups(self, legacy, fast, a, b)

    def test_empty_eligible_pool_constant_alpha_and_empty_quotes(self):
        market = Market(days=4, stocks=70)
        with ExitStack() as stack:
            market.install(stack)
            for case in ("constant", "empty_pool", "empty_quotes"):
                groups = [market.group(models=1), market.group(models=1)]
                for group in groups:
                    if case == "constant":
                        for amodel in group.alpha_models[0].models.values():
                            amodel.alpha[:] = 1.0
                    elif case == "empty_pool":
                        for port in group.benchmarks[0].ports.values():
                            port.port["secid"] += 10000
                if case == "empty_pool":
                    with self.assertRaises(KeyError):
                        run_generator(groups[0], "legacy")
                    with self.assertRaisesRegex(
                        UnsupportedGenerator, "empty candidate"
                    ):
                        run_generator(groups[1], "array")
                    with self.assertRaises(KeyError):
                        run_generator(groups[1], "auto")
                    continue
                with (
                    patch.object(
                        DATAVENDOR,
                        "get_quote_ret",
                        return_value=pd.DataFrame(
                            {"secid": np.empty(0, dtype=int), "ret": np.empty(0)}
                        ).set_index("secid"),
                    )
                    if case == "empty_quotes"
                    else nullcontext()
                ):
                    a, b = (
                        run_generator(groups[0], "legacy"),
                        run_generator(groups[1], "array"),
                    )
                    assert_groups(self, groups[0], groups[1], a, b)

    def test_auto_discards_unpublished_partial_results(self):
        market = Market(days=7)
        with ExitStack() as stack:
            market.install(stack)
            legacy, fast = market.group(), market.group()
            a = run_generator(legacy, "legacy")
            calls = 0

            def late_failure(*args):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise UnsupportedGenerator("test late unsupported data")
                return select(*args)

            with patch(
                "src.res.factor.fmp.generator_array.select", side_effect=late_failure
            ):
                b = run_generator(fast, "auto")
            assert_groups(self, legacy, fast, a, b)

    def test_publish_failure_is_not_retried_as_legacy(self):
        market = Market(days=3)
        with ExitStack() as stack:
            market.install(stack)
            group = market.group()
            with (
                patch.object(
                    group, "save_portfolios", side_effect=OSError("disk full")
                ),
                patch.object(group, "build") as legacy,
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    run_generator(group, "auto")
                legacy.assert_not_called()

    def test_multiple_nondefault_parameter_groups_use_legacy(self):
        market = Market(days=6)
        with ExitStack() as stack:
            market.install(stack)
            params = {
                "retain": dict(turn_control=0.1, indus_control=0.1),
                "rotate": dict(turn_control=0.8, indus_control=0.3, buffer_zone=0.9),
            }
            legacy, fast = (
                market.group(param_groups=params),
                market.group(param_groups=params),
            )
            a, b = (
                run_generator(legacy, "legacy"),
                run_generator(fast, "auto", block_days=2),
            )
            assert_groups(self, legacy, fast, a, b)

    def test_quote_preparation_is_block_local_and_shared_only_by_market_key(self):
        market = Market()
        with ExitStack() as stack:
            market.install(stack)
            inputs = Inputs()
            q = inputs.quote(20230102, 20230103)
            self.assertIs(inputs.quote(20230102, 20230103), q)
            self.assertIsNot(inputs.quote(20230102, 20230103, "open"), q)
            self.assertEqual(len(market.quote_calls), 2)
            inputs.clear()
            inputs.quote(20230102, 20230103)
            self.assertEqual(len(market.quote_calls), 3)
            Inputs().quote(20230102, 20230103)
            self.assertEqual(len(market.quote_calls), 4)

    def test_t50_entrypoint_and_callback_forward_only_t50_option(self):
        from src.res.factor.analytic.fmp_t50 import T50FMPTest
        from src.res.model.callback.test.detail import DetailedAlphaAnalysis
        from src.proj.bases import TestType

        market = Market(days=3)
        with ExitStack() as stack:
            market.install(stack)
            outputs = []
            for mode in ("legacy", "array", "auto"):
                group = market.group()
                universe = SimpleNamespace(get=lambda *args: group.benchmarks[0])
                factor = SimpleNamespace(
                    alpha_models=lambda: group.alpha_models,
                    universe=lambda **kw: universe,
                )
                options: dict[str, Any] = {"universe_portfolio": group.benchmarks[0]}
                if mode != "auto":
                    options["t50_backend"] = mode
                test = T50FMPTest(
                    start=int(market.dates[0]),
                    end=int(market.dates[-1]),
                    vb_level="never",
                    **options,
                )
                test.generate(cast(Any, factor))
                outputs.append(test)
            assert_groups(
                self,
                outputs[0].portfolio_group,
                outputs[1].portfolio_group,
                outputs[0].total_account,
                outputs[1].total_account,
            )
            assert_groups(
                self,
                outputs[0].portfolio_group,
                outputs[2].portfolio_group,
                outputs[0].total_account,
                outputs[2].total_account,
            )
        logger = SimpleNamespace(
            note=Mock(),
            divider=Mock(),
            subprocess=lambda **kw: nullcontext(),
            timer=lambda *a, **kw: nullcontext(),
        )
        cb = SimpleNamespace(
            fmp_tasks=[TestType.T50, TestType.TOP],
            logger=logger,
            get_factor_for_fmp_test=Mock(return_value=Mock()),
            snap_folder=None,
            config=SimpleNamespace(is_resuming=False, model_name="example"),
            trainer=SimpleNamespace(config=SimpleNamespace(beg_date=1, end_date=2)),
            indent=0,
            vb_level=0,
            t50_backend="array",
            prepared_universe_root=None,
            prepared_universe_mode="off",
            test_results={},
            test_figures={},
        )
        with (
            patch("src.res.model.callback.test.detail.FactorTestAPI.run_test") as run,
            patch("src.res.factor.util.classes.universe.Universe"),
        ):
            run.return_value.get_rslts.return_value = {}
            run.return_value.get_figs.return_value = {}
            DetailedAlphaAnalysis.fmp_test(cast(Any, cb))
            self.assertEqual(run.call_args_list[0].kwargs["t50_backend"], "array")
            self.assertNotIn("t50_backend", run.call_args_list[1].kwargs)

    def test_kernels_never_call_pandas(self):
        day = prepare_selection(
            np.arange(100), np.arange(100, dtype=float), np.arange(100) % 12
        )
        old = Position(np.array([2, 4]), np.array([0.5, 0.5]), 1)
        quotes = Quotes(np.arange(100), np.full(100, 0.01))
        conf = SimpleNamespace(
            buffer_zone=0.8, no_zone=0.5, stay_num=45, indus_slots=5, n_best=50
        )
        with ExitStack() as stack:
            for method in ("merge", "groupby", "query", "sort_values", "__init__"):
                stack.enter_context(
                    patch.object(
                        pd.DataFrame, method, side_effect=AssertionError(method)
                    )
                )
            new = select(day, old, conf, 2)
            evolved = evolve(new, 3, quotes)
            account_step(
                old, new, new, quotes, quotes, quotes, quotes, quotes, quotes, 6, 6
            )
            self.assertGreater(evolved.value, new.value)


if __name__ == "__main__":
    main()
