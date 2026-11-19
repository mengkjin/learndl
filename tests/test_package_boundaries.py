"""Keep data kernels and shared services independent of their callers."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]


class PackageBoundaryTest(unittest.TestCase):
    def test_data_and_delivery_dependency_directions(self):
        rules = [
            ('src/data/preprocess', ('src.res', 'src.automation', 'src.data.maintenance')),
            ('src/data/util/prices', ('src.data.loader', 'src.data.preprocess', 'src.data.update')),
            ('src/automation/artifact_exchange', ('src.res.rl_experimental',)),
        ]
        for directory, forbidden in rules:
            for path in (ROOT / directory).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
                    for name in names:
                        if name:
                            self.assertFalse(name.startswith(forbidden), f'{path}: {name}')

    def test_software_tests_are_outside_runtime_packages(self):
        for path in (ROOT / 'src').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        self.assertNotIn(ast.unparse(base), ('unittest.TestCase', 'TestCase'), str(path))

    def test_declared_updater_dependencies_and_cycles(self):
        from src.data.update.custom.basic import BasicCustomUpdater
        price = SimpleNamespace(DEPENDS_ON=(), UPDATE_ORDER=10)
        consumer = SimpleNamespace(DEPENDS_ON=('price',), UPDATE_ORDER=-10)
        with patch.object(BasicCustomUpdater, 'import_updaters'), patch.object(BasicCustomUpdater, 'registry', {'consumer':consumer,'price':price}):
            self.assertEqual(list(BasicCustomUpdater.iter_updaters()), [price,consumer])
            price.DEPENDS_ON = ('consumer',)
            with self.assertRaisesRegex(ValueError, 'cyclic'):
                list(BasicCustomUpdater.iter_updaters())

    def test_invalid_dependencies_fail_before_any_updater_is_yielded(self):
        from src.data.update.custom.basic import BasicCustomUpdater
        price = SimpleNamespace(DEPENDS_ON=(), UPDATE_ORDER=0)
        consumer = SimpleNamespace(DEPENDS_ON=('missing',), UPDATE_ORDER=1)
        with patch.object(BasicCustomUpdater, 'import_updaters'), patch.object(BasicCustomUpdater, 'registry', {'price': price, 'consumer': consumer}):
            with self.assertRaisesRegex(ValueError, 'Missing or cyclic'):
                next(BasicCustomUpdater.iter_updaters())

    def test_newly_ready_prices_precede_lower_priority_updates(self):
        from src.data.update.custom.basic import BasicCustomUpdater
        adj = SimpleNamespace(DEPENDS_ON=(), UPDATE_ORDER=-100)
        listing = SimpleNamespace(DEPENDS_ON=('adj',), UPDATE_ORDER=-90)
        risk = SimpleNamespace(DEPENDS_ON=(), UPDATE_ORDER=0)
        with patch.object(BasicCustomUpdater, 'import_updaters'), patch.object(BasicCustomUpdater, 'registry', {'adj': adj, 'listing': listing, 'risk': risk}):
            self.assertEqual(list(BasicCustomUpdater.iter_updaters()), [adj, listing, risk])

    def test_selective_updates_follow_declared_dependencies(self):
        from src.data.update.custom.basic import BasicCustomUpdater
        from src.data.update.updater import CustomDataUpdater
        from src.proj import Base
        events = []
        price = SimpleNamespace(__name__='price', DEPENDS_ON=(), UPDATE_ORDER=10,
                                update=Mock(side_effect=lambda **kw: events.append('price') or Base.UpdateFlag.SUCCESS))
        consumer = SimpleNamespace(__name__='consumer', DEPENDS_ON=('price',), UPDATE_ORDER=-10,
                                   update=Mock(side_effect=lambda **kw: events.append('consumer') or Base.UpdateFlag.SUCCESS))
        with patch.object(BasicCustomUpdater, 'import_updaters'), patch.object(BasicCustomUpdater, 'registry', {'price': price, 'consumer': consumer}):
            CustomDataUpdater.selective_update(['custom.consumer', 'custom.price'])
        self.assertEqual(events, ['price', 'consumer'])
