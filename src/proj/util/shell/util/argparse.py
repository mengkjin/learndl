"""
Utility functions for argparse, including parsing unknown arguments and converting to a flat dict.
"""
from __future__ import annotations
import argparse

__all__ = ['argparse_dict']

def unknown_args(unknown):
    """Build a dict from ``parse_known_args`` tail (``--a 1 --b`` or ``--a=1`` style)."""
    args = {}
    key = None
    for ua in unknown:
        if ua.startswith('--'):
            body = ua[2:]
            if '=' in body:
                eq_key, _, value = body.partition('=')
                if eq_key in args:
                    raise ValueError(f'Duplicate argument: {eq_key}')
                args[eq_key] = value if value != '' else None
                key = None
            else:
                key = body
                if key in args:
                    raise ValueError(f'Duplicate argument: {key}')
                args[key] = None
        else:
            if key is None:
                raise ValueError(f'Value without argument: {ua}')
            if args[key] is None:
                args[key] = ua
            elif isinstance(args[key] , tuple):
                args[key] = args[key] + (ua,)
            else:
                args[key] = (args[key] , ua)
    return args

def argparse_dict(**kwargs):
    """Parse known args plus ``--key value`` / ``--key=value`` pairs into a flat dict merged with ``kwargs``."""
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='py', help='Source of the script call')
    args , unknown = parser.parse_known_args()
    return args.__dict__ | unknown_args(unknown) | kwargs

