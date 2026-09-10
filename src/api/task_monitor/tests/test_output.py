from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from src.api.task_monitor.output import (
    MAX_RECORD_CHARS, OutputCache, parse_live_output, sanitize_inline_html,
)


class TaskOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.cache = OutputCache(self.root / 'cache')

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_colored_sanitizer_preserves_safe_styles_only(self) -> None:
        source = (
            '<span style="color: red; background-color: #000; position: fixed; '
            'background-image: url(bad)" onclick="bad()">safe &lt;value&gt;</span>'
            '<script>alert(1)</script>'
        )
        colored = sanitize_inline_html(source, 'colored')
        self.assertIn('color: red', colored)
        self.assertIn('background-color: #000', colored)
        self.assertNotIn('position', colored)
        self.assertNotIn('onclick', colored)
        self.assertNotIn('<script', colored)
        self.assertIn('&lt;value&gt;', colored)
        plain = sanitize_inline_html(source, 'text')
        self.assertNotIn('<span', plain)
        self.assertIn('safe <value>', plain)

    def test_live_colored_output_preserves_ansi_html_and_compacts_noise(self) -> None:
        path = self.root / 'live.md'
        path.write_text(
            '- 09:00:00.000: <span style="color: #55ff55; font-weight: bold">ok</span>\n'
            '- 09:00:00.001: ^\n- 09:00:00.002: ^\n', encoding='utf-8',
        )
        colored = parse_live_output(path, 'colored')
        self.assertIn('color: #55ff55', colored)
        self.assertIn('font-weight: bold', colored)
        self.assertIn('2 blank/control-output lines suppressed', colored)

    def test_html_cache_is_style_specific_chunked_and_survives_source_removal(self) -> None:
        path = self.root / 'report.html'
        rows = ''.join(
            f'<tr class="output-row"><td class="type-cell">STDOUT</td>'
            f'<td class="time-cell">09:00:{index:02d}</td>'
            f'<td class="content-cell"><span style="color: red">row {index}</span></td></tr>'
            for index in range(205)
        )
        path.write_text(f'<table>{rows}</table>', encoding='utf-8')
        colored = self.cache.load_page('task', path, 'colored')
        plain = self.cache.load_page('task', path, 'text')
        assert colored is not None and plain is not None
        self.assertEqual(colored.total_pages, 2)
        self.assertIn('color: red', colored.content)
        self.assertNotIn('<span', plain.content)
        self.assertNotEqual(
            self.cache._base_dir('task', path, 'colored'),
            self.cache._base_dir('task', path, 'text'),
        )
        path.unlink()
        cached_after_delete = self.cache.load_page('task', path, 'colored')
        assert cached_after_delete is not None
        self.assertTrue(cached_after_delete.cache_hit)

    def test_html_cache_marks_stderr_and_compacts_control_rows(self) -> None:
        path = self.root / 'stderr.html'
        path.write_text(
            '<table>'
            '<tr class="output-row"><td class="type-cell">STDERR</td>'
            '<td class="time-cell">09:00:00</td><td class="content-cell">^</td></tr>'
            '<tr class="output-row"><td class="type-cell">STDERR</td>'
            '<td class="time-cell">09:00:01</td><td class="content-cell">^</td></tr>'
            '<tr class="output-row"><td class="type-cell">STDERR</td>'
            '<td class="time-cell">09:00:02</td><td class="content-cell">boom</td></tr>'
            '</table>',
            encoding='utf-8',
        )
        colored = self.cache.load_page('task', path, 'colored')
        plain = self.cache.load_page('task', path, 'text')
        assert colored is not None and plain is not None
        self.assertIn('2 blank/control-output lines suppressed', colored.content)
        self.assertIn('class="output-line stderr"', colored.content)
        self.assertIn('[STDERR]', plain.content)
        self.assertNotIn('<div', plain.content)

    def test_source_change_creates_new_fingerprint(self) -> None:
        path = self.root / 'output.md'
        path.write_text('first', encoding='utf-8')
        first = self.cache.ensure('task', path, 'text')
        path.write_text('second content', encoding='utf-8')
        second = self.cache.ensure('task', path, 'text')
        self.assertNotEqual(first, second)
        assert second is not None
        self.assertFalse(any(second.parent.glob('.building-*')))

    def test_large_html_record_is_bounded(self) -> None:
        path = self.root / 'large-row.html'
        path.write_text(
            '<tr class="output-row"><td class="type-cell">STDOUT</td>'
            '<td class="time-cell">09:00:00</td><td class="content-cell">'
            + ('x' * (MAX_RECORD_CHARS * 2))
            + '</td></tr>',
            encoding='utf-8',
        )
        page = self.cache.load_page('task', path, 'text')
        assert page is not None
        self.assertLess(len(page.content), MAX_RECORD_CHARS + 100)
        self.assertIn('[record truncated]', page.content)

    def test_cleanup_applies_age_limit(self) -> None:
        path = self.root / 'output.md'
        path.write_text('content', encoding='utf-8')
        cache_dir = self.cache.ensure('task', path, 'text')
        assert cache_dir is not None
        old = time.time() - 31 * 86400
        os.utime(cache_dir / 'manifest.json', (old, old))
        result = self.cache.cleanup(max_age_days=30)
        self.assertEqual(result['removed_entries'], 1)
        self.assertFalse(cache_dir.exists())

    def test_cleanup_applies_lru_size_limit(self) -> None:
        paths = [self.root / f'output-{index}.md' for index in range(2)]
        for index, path in enumerate(paths):
            path.write_text(f'content {index}', encoding='utf-8')
            cache_dir = self.cache.ensure(f'task-{index}', path, 'text')
            assert cache_dir is not None
            accessed = time.time() - (2 - index) * 60
            os.utime(cache_dir / 'manifest.json', (accessed, accessed))
        result = self.cache.cleanup(max_bytes=1, max_age_days=30)
        self.assertEqual(result['removed_entries'], 2)
        self.assertEqual(result['remaining_bytes'], 0)

    def test_cleanup_removes_abandoned_atomic_build(self) -> None:
        building = self.cache.root / 'text' / 'task-key' / '.building-abandoned'
        building.mkdir(parents=True)
        (building / 'chunk.tmp').write_text('partial', encoding='utf-8')
        old = time.time() - 7200
        os.utime(building, (old, old))
        result = self.cache.cleanup()
        self.assertEqual(result['removed_entries'], 1)
        self.assertFalse(building.exists())
