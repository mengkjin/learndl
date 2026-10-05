"""Research comparison prompts must complete before any calculation starts."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.api.calls.model_compare import CompareModels, comparison_schema, parse_parameters
from src.proj.util.cli.ask import AskFlag
from src.res.model.analytic.compare import CompareConfig


class CompareCliTest(unittest.TestCase):
    def test_default_and_custom_parameters(self):
        values = comparison_schema().signature_defaults
        config, replica, output = parse_parameters(values)
        self.assertEqual(config, CompareConfig())
        self.assertEqual(replica, 0)
        self.assertIsNone(output)
        config, replica, output = parse_parameters(dict(
            values, start=20250101, end=20251231, periods='all, quarter',
            custom_periods='{"holdout": [20250601, 20251231]}',
            analyze_pred=True, hidden_model_num=2, output_dir='/tmp/compare-test',
        ))
        self.assertEqual(config.periods, ('all', 'quarter'))
        self.assertTrue(config.analyze_pred)
        self.assertEqual(replica, 2)
        with self.assertRaises(ValueError):
            parse_parameters(dict(values, start=20260101, end=20250101))

    def test_selection_retry_configuration_and_cancel(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ('a', 'b'):
                    report = root / name / 'results/detailed_alpha_data.xlsx'
                    report.parent.mkdir(parents=True)
                    report.touch()
                schema = comparison_schema()
                events = []
                selections = iter([[1], [1, 2]])
                def select(*args, **kwargs):
                    self.assertEqual(args[0], ['a', 'b'])
                    events.append('select')
                    return AskFlag('valid').set_result(next(selections))
                def configure(*args, **kwargs):
                    events.append('configure')
                    return (AskFlag('exit') if cancel else
                            AskFlag('valid').set_result([schema.signature_defaults]))
                def calculate(models, **kwargs):
                    events.append('calculate')
                    self.assertEqual([m.model for m in models], [root / 'a', root / 'b'])
                    self.assertEqual(kwargs['config'], CompareConfig())
                    return SimpleNamespace(output_paths={})
                with patch('src.api.calls.model_compare.PATH', SimpleNamespace(model=root)), \
                     patch('src.api.calls.model_compare.Options.available_models', return_value=['a', 'b', 'missing']), \
                     patch('src.api.calls.model_compare.comparison_schema', return_value=schema), \
                     patch('src.api.calls.model_compare.AskFor.Selections', side_effect=select), \
                     patch('src.api.calls.model_compare.AskFor.ScriptKwargs', side_effect=configure), \
                     patch('src.res.model.analytic.compare.compare_models', side_effect=calculate) as compute:
                    CompareModels().run()
                self.assertEqual(events, ['select', 'select', 'configure'] + ([] if cancel else ['calculate']))
                self.assertEqual(compute.call_count, 0 if cancel else 1)

    def test_research_menu_dispatch(self):
        from src.api.calls.launcher import DirectCallHub
        self.assertIn(CompareModels, [entry[1] for entry in DirectCallHub._research_entries()])
        with patch.object(DirectCallHub, '_pick_direct_call', return_value=CompareModels), \
             patch.object(CompareModels, 'spawn_in_pane') as spawn:
            DirectCallHub()._dispatch_top_level('Research Operations')
        spawn.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
