"""Low-level utils: timing, silence, colored terminal I/O, singletons, HTML helpers, ``Once``."""
from .func import as_int_array
from .types import strPath , strPaths , StrEnum , lit
from .duration import Duration
from .silence import Silence
from .output import stdout , stderr , FormatStr
from .singleton import SingletonMeta , SingletonABCMeta , NoInstanceMeta
from .htmls import str_to_html , replace_ansi_sequences , ansi_codes_to_span , figure_to_base64 , dataframe_to_html , figure_to_html
from .once import Once