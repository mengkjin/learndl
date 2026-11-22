"""Persist shared console card visibility independently of service state."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import portalocker

__all__ = ['Preferences']


class Preferences:
    """Store a versioned set of hidden service IDs on the console server."""

    def __init__(self, state: Path, services):
        self.path = state / 'preferences.json'
        self.services = frozenset(services)

    def read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if (not isinstance(data, dict) or type(data.get('version')) is not int
                    or data['version'] != 1 or not isinstance(data.get('hidden'), list)
                    or not all(isinstance(key, str) for key in data['hidden'])):
                raise ValueError('Invalid preferences format')
            return {'version': 1, 'hidden': sorted(self.services.intersection(data['hidden']))}
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as exc:
            logging.getLogger(__name__).warning('Could not read console preferences: %s', exc)
        return {'version': 1, 'hidden': []}

    def update(self, data: dict) -> dict:
        show_all = data == {'show_all': True} and data['show_all'] is True
        service = data.get('service')
        visible = data.get('visible')
        if not show_all and (data.keys() != {'service', 'visible'} or not isinstance(service, str)
                             or service not in self.services or type(visible) is not bool):
            raise ValueError('Expected a known service ID and boolean visibility, or show_all: true')
        with portalocker.Lock(self.path.with_suffix('.lock'), timeout=3):
            hidden = set(self.read()['hidden'])
            if show_all:
                hidden.clear()
            elif visible:
                hidden.discard(service)
            else:
                hidden.add(service)
            result = {'version': 1, 'hidden': sorted(hidden)}
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
