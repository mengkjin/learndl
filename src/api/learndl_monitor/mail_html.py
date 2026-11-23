"""Render archived HTML as static mail content without active embeds or links."""
from __future__ import annotations

from html import escape
from html.parser import HTMLParser
import re

__all__ = ['static_mail_html']

_TAGS = frozenset('body style article section div span p br hr blockquote pre code kbd samp h1 h2 h3 h4 h5 h6 '
                  'b strong i em u s small sub sup mark ul ol li dl dt dd table caption colgroup col '
                  'thead tbody tfoot tr th td details summary img'.split())
_VOID = {'br', 'hr', 'col', 'img'}
_DROP_CONTENT = {'title', 'script', 'template', 'svg', 'math', 'object', 'iframe', 'noscript'}
_DATA_IMAGE = re.compile(r'data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=\s]+', re.I)
_TABLE_ATTRS = {
    'table': {'border', 'cellpadding', 'cellspacing', 'align'},
    'colgroup': {'span'}, 'col': {'span', 'align', 'valign'},
    'tr': {'align', 'valign'}, 'th': {'colspan', 'rowspan', 'align', 'valign', 'nowrap'},
    'td': {'colspan', 'rowspan', 'align', 'valign', 'nowrap'},
}
_LOG_COLUMNS = """<style>
table.output-table > thead > tr > :is(.col-index, .col-type, .col-vb-level, .col-time),
table.output-table > tbody > tr > :is(.index-cell, .type-cell, .vb-level-cell, .time-cell) {
    white-space: nowrap;
    overflow-wrap: normal;
    word-break: normal;
}
</style>"""


class _StaticMail(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.root = '<html>'
        self.skip = ''
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        if self.skip:
            return
        if tag in _DROP_CONTENT:
            self.skip = tag
            return
        if tag not in _TAGS and tag != 'html':
            return
        if tag == 'style':
            self.in_style = True
        attributes = []
        for key, value in attrs:
            if (key in {'class', 'id', 'style', 'title', 'alt', 'width', 'height', 'lang', 'dir'}
                    or key in _TABLE_ATTRS.get(tag, ())
                    or tag == 'img' and key == 'src' and _DATA_IMAGE.fullmatch(value or '')):
                attributes.append(f' {key}="{escape(value or "", quote=True)}"')
        opening = f'<{tag}{"".join(attributes)}>'
        if tag == 'html':
            self.root = opening
        else:
            self.output.append(opening)

    def handle_startendtag(self, tag, attrs):
        if self.skip or tag in _DROP_CONTENT:
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.skip:
            if self.skip == tag:
                self.skip = ''
        elif tag in _TAGS and tag not in _VOID:
            if tag == 'style':
                self.in_style = False
            self.output.append(f'</{tag}>')

    def handle_data(self, data):
        if self.in_style:
            # HTMLParser and browser recovery can disagree on malformed end tags.
            self.output.append(re.sub(r'</style', lambda _: r'\3c /style', data, flags=re.I))
        elif not self.skip:
            self.output.append(escape(data))


def static_mail_html(source: str) -> str:
    """Keep formatting and embedded raster images while removing navigation and active content."""
    parser = _StaticMail()
    parser.feed(source)
    parser.close()
    return ("<!doctype html>" + parser.root + "<meta charset=utf-8><meta http-equiv=Content-Security-Policy "
            '''content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; '''
            '''base-uri 'none'; form-action 'none'">'''
            + _LOG_COLUMNS + ''.join(parser.output) + '</html>')
