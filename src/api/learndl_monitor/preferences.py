"""Persist shared console card visibility and order."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import portalocker

__all__ = ['Preferences']


class Preferences:
    """Store card preferences on the console server."""

    def __init__(self, state: Path, services):
        self.path = state / 'preferences.json'
        self.services = tuple(services)

    def read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if (not isinstance(data, dict) or type(data.get('version')) is not int
                    or data['version'] != 1 or not isinstance(data.get('hidden'), list)
                    or not all(isinstance(key, str) for key in data['hidden'])):
                raise ValueError('Invalid preferences format')
            order = data.get('order', [])
            if not isinstance(order, list):
                order = []
            order = list(dict.fromkeys(key for key in order if isinstance(key, str) and key in self.services))
            order.extend(key for key in self.services if key not in order)
            return {'version': 1, 'hidden': sorted(set(self.services).intersection(data['hidden'])), 'order': order}
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as exc:
            logging.getLogger(__name__).warning('Could not read console preferences: %s', exc)
        return {'version': 1, 'hidden': [], 'order': list(self.services)}

    def update(self, data: dict) -> dict:
        show_all = data == {'show_all': True} and data['show_all'] is True
        service = data.get('service')
        visible = data.get('visible')
        move = data.get('move')
        valid_service = isinstance(service, str) and service in self.services
        visibility_change = data.keys() == {'service', 'visible'} and type(visible) is bool
        reorder = data.keys() == {'service', 'move'} and move in ('earlier', 'later')
        if not show_all and not (valid_service and (visibility_change or reorder)):
            raise ValueError('Expected a service visibility or move operation, or show_all: true')
        with portalocker.Lock(self.path.with_suffix('.lock'), timeout=3):
            current = self.read()
            hidden, order = set(current['hidden']), current['order']
            if show_all:
                hidden.clear()
            elif reorder:
                visible_positions = [i for i, key in enumerate(order) if key not in hidden]
                position = order.index(service)
                if position not in visible_positions:
                    raise ValueError('The service is hidden')
                neighbor = visible_positions.index(position) + (-1 if move == 'earlier' else 1)
                if 0 <= neighbor < len(visible_positions):
                    other = visible_positions[neighbor]
                    order[position], order[other] = order[other], order[position]
            elif visible:
                hidden.discard(service)
            else:
                hidden.add(service)
            result = {'version': 1, 'hidden': sorted(hidden), 'order': order}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.preferences-', delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(result, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    temporary.replace(self.path)
                finally:
                    temporary.unlink(missing_ok=True)
            return result
