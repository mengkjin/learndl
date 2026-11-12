"""Exercise the comparison form using portable temporary Excel reports."""
from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

__all__ = []


class CompareAppTest(unittest.TestCase):
    def test_form_comparison_and_language_switch_preserve_result(self):
        fixture = importlib.import_module('test_model_compare').CompareInputTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from src.api.learndl_monitor import compare_app
        original = compare_app.calculate

        def offline(sources, parameters):
            return original(sources, dict(parameters, trade_dates=tuple(fixture.dates)))

        with patch.object(compare_app, 'calculate', side_effect=offline) as calculate:
            app = AppTest.from_string('from src.api.learndl_monitor.compare_app import main\nmain()', default_timeout=30).run()
            self.assertFalse(app.exception)
            app.text_area[0].set_value('\n'.join(str(p) for p in fixture.paths))
            next(b for b in app.button if '比较' in b.label).click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.dataframe)
            count = calculate.call_count
            app.selectbox[0].select('English').run()
            self.assertFalse(app.exception)
            self.assertEqual(calculate.call_count, count)
            self.assertTrue(app.dataframe)


if __name__ == '__main__':
    unittest.main()
