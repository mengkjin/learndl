"""Export DataFrames to Excel and matplotlib figures to a single PDF."""
from __future__ import annotations
import os

from src.proj.log import Logger
from src.proj.core import strPath , lit
from src.proj.util.functional.mpl_config import configure_matplotlib

__all__ = ['figs_to_pdf']

_NONCHAR_STRIP = str.maketrans({"\ufeff": "", "\ufffe": "", "\uffff": ""})

def _sanitize_figure_text(fig) -> None:
    """Strip non-characters from all text objects in figure.

    Matplotlib may warn when rendering non-characters like U+FFFE.
    """
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    assert isinstance(fig , Figure) , f'fig must be a matplotlib figure , but got {type(fig)}'
    for obj in fig.findobj(match=Text):
        try:
            s = obj.get_text()
        except Exception:
            continue
        if isinstance(s, str) and any(ch in s for ch in ("\ufeff", "\ufffe", "\uffff")):
            obj.set_text(s.translate(_NONCHAR_STRIP))

def figs_to_pdf(
    figs , path : strPath , prefix : str | None = None , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 3 ,
    close : bool = True ,
) -> bool:
    """Save figures to one PDF and optionally close each figure.

    Uses the process Agg default so this is safe on ``AsyncSaver`` worker threads.
    Pass ``close=False`` if the caller must retain live figures (e.g. interactive GUI).

    Returns:
        True when the PDF was written.
    """
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    with PdfPages(path) as pdf:
        for key, fig in figs.items():
            _sanitize_figure_text(fig)
            pdf.savefig(fig)
            if close:
                plt.close(fig)
    if prefix: 
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return True