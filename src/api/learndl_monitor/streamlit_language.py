"""Language selection shared by the Streamlit services opened from the console."""
from __future__ import annotations

from collections.abc import Callable
from typing import overload

import streamlit as st

__all__ = ['text', 'page_toolbar']

def text(zh, en):
    return zh if st.session_state.get('monitor_language') == 'zh' else en


@overload
def page_toolbar[T](*, default: str = 'zh', render_settings: Callable[[], T]) -> tuple[bool, T]: ...


@overload
def page_toolbar(*, default: str = 'zh', render_settings: None = None) -> tuple[bool, None]: ...


def page_toolbar(*, default='zh', render_settings=None):
    """Render the settings and refresh controls used by Streamlit child services."""
    names = {'zh': '中文', 'en': 'English'}
    initial = st.query_params.get('lang')
    if initial in names:
        st.session_state['monitor_language_choice'] = initial
        del st.query_params['lang']
    st.session_state.setdefault('monitor_language_choice', default)
    st.session_state['monitor_language'] = st.session_state['monitor_language_choice']
    st.html("""
        <style>
        :is(.st-key-monitor-page-toolbar [data-testid="stPopoverButton"], .st-key-monitor-page-refresh button) {
            width:40px;min-height:40px;padding:0;border:1px solid #3b4b60;
            border-radius:7px;background:#202c3a;color:#e2e8f0;
        }
        :is(.st-key-monitor-page-toolbar [data-testid="stPopoverButton"], .st-key-monitor-page-refresh button):is(:hover,:focus-visible) {border-color:#55ccb9;}
        .st-key-monitor-page-toolbar [data-testid="stPopoverButton"] > div > :last-child {
            display:none;
        }
        :is(.st-key-monitor-page-toolbar [data-testid="stPopoverButton"], .st-key-monitor-page-refresh button)
        [data-testid="stMarkdownContainer"] {
            position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);
        }
        [data-testid="stPopoverBody"]:has(.st-key-monitor-settings-fields) {
            box-sizing:border-box;
            width:min(23rem,calc(100vw - 28px));
            min-width:0 !important;max-width:calc(100vw - 28px) !important;
            max-height:65dvh;overflow:auto;
            border-radius:10px;background:#17212c;
        }
        </style>
    """)
    with st.container(key='monitor-page-toolbar', horizontal=True, horizontal_alignment='right', gap='xxsmall'):
        settings_label = text('设置', 'Settings')
        with st.popover(settings_label, icon=':material/settings:', help=settings_label,
                        key='monitor-page-settings'):
            with st.container(key='monitor-settings-fields'):
                st.selectbox(text('页面语言', 'Page language'), list(names), format_func=names.__getitem__,
                             key='monitor_language_choice')
                settings = render_settings() if render_settings else None
        refresh_label = text('刷新', 'Refresh')
        refresh = st.button(refresh_label, icon=':material/refresh:', help=refresh_label, key='monitor-page-refresh')
    return refresh, settings
