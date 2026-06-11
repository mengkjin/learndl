"""
CLI entry point for batch preprocessing of all registered ``PrePro_*`` processors.

Usage (from project root)::

    python -m src.data.preprocess.task               # fit mode
    python -m src.data.preprocess.task --predict 1   # predict mode
"""
from __future__ import annotations

from typing import Any , Iterable

from src.proj import Base , Dates
from .processors import PrePros

__all__ = ['PreProcessorTask']

DATASET_FIT : tuple[str,...] = tuple(PrePros.keys())
DATASET_PREDICT : tuple[str,...] = DATASET_FIT

class PreProcessorTask(Base.BasicUpdater):
    """
    Batch runner that iterates over all registered preprocessors and calls
    ``PreProcessor.update()``.

    Called from scripts or the CLI; uses argparse for the ``--confirm`` flag.
    """
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK , Base.UpdateType.RECALC)
    
    @classmethod
    def parse_update_input(
        cls , update_type : Base.UpdateType , rollback_date : int | None = None , 
        frame : Base.lit.DataBlockTimeFrame = 'fit', confirm : bool = True , **kwargs) -> dict[str , Any]:
        return {
            'reconstruct' : update_type == Base.UpdateType.RECALC ,
            'rollback_date' : rollback_date if update_type == Base.UpdateType.ROLLBACK else None ,
            'confirm' : False if frame == 'predict' else confirm ,
            'start' : 20001001 if update_type == Base.UpdateType.RECALC else None ,
            'end' : 20991231 if update_type == Base.UpdateType.RECALC else None ,
        }

    @classmethod
    def proceed_update(
        cls , * , reconstruct : bool = False , rollback_date : int | None = None , 
        frame : Base.lit.DataBlockTimeFrame = 'fit', confirm : bool = True , data_types : Iterable[str] | None = None , 
        indent : int = 0 , vb_level : Any = 1 , force_update : bool = False , **kwargs) -> Base.UpdateFlag:
        """
        Run the preprocessing update for all (or selected) registered processors.

        Parameters
        ----------
        frame : Base.lit.DataBlockTimeFrame
            Use ``'predict'`` mode; otherwise use ``'fit'`` mode.
        confirm : bool
            If True, will prompt for confirmation.
        parser : argparse.ArgumentParser | None
            Pre-built parser (used when called from a parent script).
        data_types : list[str] | None
            Explicit list of processor keys to update.  Defaults to all.
        force_update : bool
            If True, skip the "already updated" check.
        """
        if confirm: 
            from src.proj.util.functional.ask import AskFor
            flag = AskFor.Confirmation(title = 'Are you sure to update the preprocessed data?')
            if not flag.yes:
                return Base.UpdateFlag.FAILED
        
        data_types = list(data_types) if data_types else (list(DATASET_PREDICT) if frame == 'predict' else list(DATASET_FIT))
        
        cls.logger.note(f'Data PreProcessing for {"fitting" if frame == 'fit' else "predicting"} start with {data_types} datas !')
        cls.logger.stdout(f'Will process {data_types} from {Dates(PrePros.start_date(frame = frame))}' , idt = 1 , vb = 1)

        flags = Base.UpdateFlagList()
        for data_type in data_types:
            proc = PrePros.get_processor(data_type , frame = frame , indent = cls.logger.indent + 1 , vb_level = cls.logger.vb_level + 1)
            flags += proc.build(force_build = force_update , confirm = False , reconstruct = reconstruct , rollback_date = rollback_date)
            cls.logger.divider(vb = 3)

        return flags.summarize()

    @classmethod
    def update(
        cls , * , 
        frame : Base.lit.DataBlockTimeFrame = 'fit', confirm : bool = True , data_types : list[str] | None = None , 
        indent : int = 0 , vb_level : Any = 1 , force_update : bool = False , **kwargs
    ) -> Base.UpdateFlag:
        """
        update the preprocessed data
        Args:
            frame : Base.lit.DataBlockTimeFrame
                Use ``'predict'`` mode; otherwise use ``'fit'`` mode.
            confirm : bool
                If True, will prompt for confirmation.
            data_types : list[str] | None
                Explicit list of processor keys to update.  Defaults to all.
            indent : int
                Indent level for logging.
            vb_level : Any
                Verbosity level for logging.
            force_update : bool
                If True, skip the "already updated" check.
            kwargs : dict
                Additional keyword arguments.
        Returns:
            Base.UpdateFlag
                The update flag.
        """
        return super().update(frame = frame , confirm = confirm , data_types = data_types , indent = indent , vb_level = vb_level , force_update = force_update , **kwargs)

    @classmethod
    def reconstruct(
        cls , * , 
        frame : Base.lit.DataBlockTimeFrame = 'fit', confirm : bool = True , data_types : list[str] | None = None , 
        indent : int = 0 , vb_level : Any = 1 , force_update : bool = False , **kwargs
    ) -> Base.UpdateFlag:
        """
        Alias for ``PreProcessorTask.recalculate``.
        Args:
            frame : Base.lit.DataBlockTimeFrame
                Use ``'predict'`` mode; otherwise use ``'fit'`` mode.
            confirm : bool
                If True, will prompt for confirmation.
            data_types : list[str] | None
                Explicit list of processor keys to update.  Defaults to all.
        """
        return cls.recalculate(
            frame = frame , confirm = confirm , data_types = data_types , 
            indent = indent , vb_level = vb_level , force_update = force_update , **kwargs)