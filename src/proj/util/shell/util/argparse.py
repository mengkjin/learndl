from __future__ import annotations
import argparse

__all__ = ['argparse_dict']

def unknown_args(unknown):
    """Build a dict from ``parse_known_args`` tail (``--a 1 --b`` style)."""
    args = {}
    for ua in unknown:
        if ua.startswith('--'):
            key = ua[2:]
            if key not in args:
                args[key] = None
            else:
                raise ValueError(f'Duplicate argument: {key}')
        else:
            if args[key] is None:
                args[key] = ua
            elif isinstance(args[key] , tuple):
                args[key] = args[key] + (ua,)
            else:
                args[key] = (args[key] , ua)
    return args

def argparse_dict(**kwargs):
    """Parse known args plus ``--key value`` pairs into a flat dict merged with ``kwargs``."""
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='py', help='Source of the script call')
    args , unknown = parser.parse_known_args()
    return args.__dict__ | unknown_args(unknown) | kwargs

