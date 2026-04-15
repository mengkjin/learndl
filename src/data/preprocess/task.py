"""
CLI entry point for batch preprocessing of all registered ``PrePro_*`` processors.

Usage (from project root)::

    python -m src.data.preprocess.task               # fit mode
    python -m src.data.preprocess.task --predict 1   # predict mode
"""
import argparse

from typing import Any

from src.proj import Logger , Dates , Proj

from .processors import PrePros

__all__ = ['PreProcessorTask']

DATASET_FIT = [*PrePros.keys()]
DATASET_PREDICT = DATASET_FIT

class PreProcessorTask:
    """
    Batch runner that iterates over all registered preprocessors and calls
    ``PreProcessor.update()``.

    Called from scripts or the CLI; uses argparse for the ``--confirm`` flag.
    """
    @classmethod
    def update(cls , predict = False, confirm = 0 , * , parser = None , data_types : list[str] | None = None , indent : int = 0 , vb_level : Any = 1 ,
               force_update : bool = False):
        """
        Run the preprocessing update for all (or selected) registered processors.

        Parameters
        ----------
        predict : bool
            If True, use ``'predict'`` mode; otherwise use ``'fit'`` mode.
        confirm : int
            Non-zero value skips the interactive confirmation prompt.
        parser : argparse.ArgumentParser | None
            Pre-built parser (used when called from a parent script).
        data_types : list[str] | None
            Explicit list of processor keys to update.  Defaults to all.
        force_update : bool
            If True, skip the "already updated today" check.
        """
        vb_level = Proj.vb(vb_level)
        if parser is None:
            parser = argparse.ArgumentParser(description = 'manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()
        if not predict and not args.confirm and \
            not input('Confirm update data? type "yes" to confirm!').lower()[0] == 'y' : 
            return
        
        if data_types:
            keys = data_types
        else:
            keys = DATASET_PREDICT if predict else DATASET_FIT
            
        Logger.note(f'Data PreProcessing for {"fitting" if not predict else "predicting"} start with {keys} datas !' , indent = indent , vb_level = vb_level)
        Logger.stdout(f'Will process {keys} from {Dates(PrePros.start_date(type = 'fit' if not predict else 'predict'))}' , indent = indent + 1 , vb_level = vb_level + 1)

        for key in keys:
            proc = PrePros.get_processor(key , type = 'fit' if not predict else 'predict')
            proc.update(force_update = force_update , indent = indent + 1 , vb_level = vb_level + 1)
            Logger.divider(vb_level = vb_level + 3)