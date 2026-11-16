import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.res.factor.calculator.backfill import fill_calculator, run_backfill


class Calculator:
    factor_name = 'example'
    meta_type = 'stock'
    level = 'level0'

    def __init__(self):
        self.values = {20120103, 20120104}
        self.stats = {'daily': {20110104}, 'weekly': set()}
        self.calls = []

    def target_dates(self, start, end, overwrite=False):
        assert not overwrite
        return sorted(d for d in self.values if start <= d <= end)

    def update_all_factors(self, start, end, overwrite=False):
        assert start == end and not overwrite
        self.calls.append(('value', start))
        self.values.remove(start)
        for dates in self.stats.values():
            dates.add(start)

    def stats_target_dates(self, start, end, overwrite=False):
        assert not overwrite
        return {key: sorted(d for d in dates if start <= d <= end) for key, dates in self.stats.items()}

    def update_periodic_stats(self, period, batch, overwrite=False):
        assert not overwrite
        self.calls.append((period, batch))
        self.stats[period].difference_update(batch)


class HistoryBackfillTest(unittest.TestCase):
    def test_full_history_stats_new_values_batches_and_resume(self):
        calc = Calculator()
        output = io.StringIO()
        with redirect_stdout(output):
            fill_calculator(calc, 20110101, 20261116, batch_size=2)
            count = len(calc.calls)
            fill_calculator(calc, 20110101, 20261116, batch_size=2)
        self.assertEqual(len(calc.calls), count)
        self.assertFalse(calc.values)
        self.assertTrue(all(not dates for dates in calc.stats.values()))
        self.assertIn('example date=20120103', output.getvalue())
        self.assertIn('20110104', output.getvalue())
        self.assertTrue(all(len(batch) <= 2 for kind, batch in calc.calls if kind != 'value'))

    def test_dry_run_does_not_write(self):
        calc = Calculator()
        with redirect_stdout(io.StringIO()):
            fill_calculator(calc, 20110101, 20261116, dry_run=True)
        self.assertEqual(calc.calls, [])

    def test_unsaved_value_fails(self):
        calc = Calculator()
        with patch.object(calc, 'update_all_factors'), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'example@20120103'):
                fill_calculator(calc, 20110101, 20261116)

    def test_dependency_order_and_serial_scope(self):
        import src.res.factor.calculator.backfill as module
        from src.proj.util.functional.parallel import parallel
        import threading
        first, second = Calculator(), Calculator()
        second.level = 'level1'
        order = []
        owner = threading.get_ident()
        def fill(calc, *args):
            order.append(calc.level)
            self.assertEqual(set(parallel([threading.get_ident] * 3).values()), {owner})
        with patch.object(module.CALENDAR, 'updated', return_value=20261116), patch.object(module.FactorCalculator, 'iter', return_value=[second, first]), patch.object(module, 'fill_calculator', side_effect=fill), redirect_stdout(io.StringIO()):
            run_backfill(20110101, 20261116)
        self.assertEqual(order, ['level0', 'level1'])
