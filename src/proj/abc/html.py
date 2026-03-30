import re
import html
import base64
import io
import pandas as pd
from typing import Any
from matplotlib.figure import Figure

_palette : dict[int , str] = {
    30: 'black',
    31: 'red',
    32: 'green',
    33: 'yellow',
    34: 'blue',
    35: 'magenta',
    36: 'cyan',
    37: 'white',
    90: '#7f7f7f',      # bright black / gray
    91: '#ff5555',      # bright red
    92: '#55ff55',      # bright green
    93: '#ffff55',      # bright yellow
    94: '#5555ff',      # bright blue
    95: '#ff55ff',      # bright magenta
    96: '#55ffff',      # bright cyan
    97: '#ffffff',      # bright white
}

def str_to_html(text: str | Any):
    """capture string to html"""
    
    assert isinstance(text, str) , f"text must be a string , but got {type(text)}"

    text = html.escape(text)
    text = re.sub(r'(?:\u001b\[[\d;]*m)+', replace_ansi_sequences, text)
    
    return text

def replace_ansi_sequences(match):
    """replace ANSI sequences to html span tag"""
    # match.group(0) contains all continuous ANSI sequences
    sequences = match.group(0)
    all_codes = []

    for seq_match in re.finditer(r'\u001b\[([\d;]*)m', sequences):
        codes_str = seq_match.group(1)
        if codes_str:
            all_codes.extend(codes_str.split(';'))
    
    return ansi_codes_to_span(all_codes)

def ansi_codes_to_span(codes):
    """convert ANSI codes list to a single span tag"""
    styles = []
    has_fg_color = False
    has_bg_color = False
    for code_str in codes:
        if not code_str:
            continue
        code = int(code_str)
        if code == 0:
            return '</span>'
        elif code == 1:
            styles.append('font-weight: bold')
        elif code == 3:
            styles.append('font-style: italic')
        elif code in _palette:
            # foreground colors
            styles.append(f'color: {_palette[code]}')
            has_fg_color = True
        elif code - 10 in _palette:
            # background colors
            styles.append(f'background-color: {_palette[code - 10]}')
            has_bg_color = True
    if not has_fg_color and has_bg_color:
        styles.append('color: white')
    
    if styles:
        return f'<span style="{"; ".join(styles)};">'
    return ''

def dataframe_to_html(df: pd.DataFrame | pd.Series | Any):
    """capture display object (dataframe or other object)"""
    assert isinstance(df, (pd.DataFrame , pd.Series)) , f"obj must be a dataframe or series , but got {type(df)}"
    try:
        # get dataframe html representation
        html_table = getattr(df , '_repr_html_')() if hasattr(df, '_repr_html_') else df.to_html(classes='dataframe')
        content = f'<div class="dataframe">{html_table}</div>'
    except Exception:
        # downgrade to text format
        content = f'<div class="df-fallback"><pre>{html.escape(df.to_string())}</pre></div>'
    return content

def figure_to_base64(fig : Figure | Any):
    """convert matplotlib figure to base64 string"""
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    except Exception as e:
        from src.proj import Logger
        Logger.error(f"Error converting figure to base64: {e}")
        return None
    
def figure_to_html(fig: Figure | Any):
    """capture matplotlib figure"""
    assert isinstance(fig, Figure) , f"fig must be a matplotlib figure , but got {type(fig)}"
    content = None
    try:
        if fig.get_axes():  # check if figure has content
            if image_base64 := figure_to_base64(fig):
                content = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto; border-radius: 5px; margin: 2px 0;">'
    except Exception as e:
        from src.proj import Logger
        Logger.error(f"Error capturing matplotlib figure: {e}")
        content = f'<div class="figure-fallback"><pre>Error capturing matplotlib figure: {e}</pre></div>'
    return content
