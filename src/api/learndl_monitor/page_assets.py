"""Shared browser assets for the native monitor pages."""
from __future__ import annotations

from pathlib import Path

__all__ = ['PAGE_ASSETS', 'serve_page_asset']

PAGE_ASSETS = {
    '/monitor-navigation.js': ('navigation.js', 'text/javascript'),
    '/monitor-toolbar.css': ('toolbar.css', 'text/css'),
}


def serve_page_asset(handler, path):
    filename, mime = PAGE_ASSETS[path]
    content = Path(__file__).with_name(filename).read_bytes()
    handler.send_response(200)
    handler.send_header('Content-Type', mime + '; charset=utf-8')
    handler.send_header('Content-Length', str(len(content)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(content)
