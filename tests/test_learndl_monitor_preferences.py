"""Exercise shared console preferences across requests and restarts."""
from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.api.learndl_monitor.preferences import Preferences

__all__ = []


class PreferencesTest(unittest.TestCase):
    def test_reorder_visible_cards_preserves_hidden_slots_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            first = Preferences(state, ('tasks', 'storage', 'compare'))
            first.path.write_text('{"version":1,"hidden":["storage"]}')
            self.assertEqual(first.read()['order'], ['tasks', 'storage', 'compare'])
            self.assertEqual(first.update({'service': 'compare', 'move': 'earlier'})['order'],
                             ['compare', 'storage', 'tasks'])
            restarted = Preferences(state, ('tasks', 'storage', 'compare', 'research'))
            self.assertEqual(restarted.read()['order'], ['compare', 'storage', 'tasks', 'research'])
            self.assertEqual(restarted.read()['hidden'], ['storage'])
            self.assertEqual(restarted.update({'service': 'compare', 'move': 'later'})['order'],
                             ['tasks', 'storage', 'compare', 'research'])
            with self.assertRaisesRegex(ValueError, 'hidden'):
                restarted.update({'service': 'storage', 'move': 'earlier'})

    def test_concurrent_changes_survive_restart_and_allow_new_services(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            first = Preferences(state, ('tasks', 'compare'))
            self.assertEqual(first.read()['hidden'], [])
            with ThreadPoolExecutor(max_workers=2) as workers:
                list(workers.map(lambda service: first.update({'service': service, 'visible': False}),
                                 ('tasks', 'compare')))
            restarted = Preferences(state, ('tasks', 'compare', 'research'))
            self.assertEqual(restarted.read()['hidden'], ['compare', 'tasks'])
            self.assertEqual(restarted.update({'service': 'tasks', 'visible': True})['hidden'], ['compare'])
            self.assertEqual(restarted.update({'show_all': True})['hidden'], [])
            self.assertEqual(first.read()['hidden'], [])

    def test_invalid_updates_and_corrupt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            preferences = Preferences(Path(directory), ('tasks',))
            preferences.update({'service': 'tasks', 'visible': False})
            for invalid in ({'show_all': 1}, {'service': 'unknown', 'visible': False},
                            {'service': 'tasks', 'visible': 'false'}):
                with self.assertRaises(ValueError):
                    preferences.update(invalid)
            self.assertEqual(preferences.read()['hidden'], ['tasks'])
            preferences.path.write_text('malformed JSON')
            with self.assertLogs('src.api.learndl_monitor.preferences', level='WARNING'):
                self.assertEqual(preferences.read()['hidden'], [])
