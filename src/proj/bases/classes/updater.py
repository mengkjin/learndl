"""basic updater class for the project"""

from __future__ import annotations
from typing import Any

from src.proj.core.types import lit
from src.proj.cal import CALENDAR
from src.proj.bases.enums import UpdateType , UpdateFlag

from .bound_logger import BoundLogger

__all__ = ['BasicUpdater']

class BasicUpdater(BoundLogger):
    """
    Basic updater class for the project.
    Must implement 'proceed_update' and 'parse_update_input' methods.
    'proceed_update' should take the update parameters and return UpdateFlag.
    'parse_update_input' should parse the update parameters and return a dictionary of update parameters.
    """
    UPDATE_ALIAS : str = 'update'
    ACCEPTABLE_UPDATE_TYPES : tuple[UpdateType,...] = (UpdateType.UPDATE, )
    START_DATE : int = 19000101

    @property
    def name(self) -> str:
        """name of the updater"""
        return self.__class__.__name__

    @classmethod
    def parse_update_input(
        cls , update_type : UpdateType , key: lit.DataUpdateKey | None = None , 
        rollback_date : int | None = None , start : int | None = None , end : int | None = None , **kwargs) -> dict[str , Any]:
        """parse the update parameters"""
        if update_type == UpdateType.UPDATE:
            start , end = CALENDAR.update_schedule(cls.START_DATE , key=key)
            overwrite = False
        elif update_type == UpdateType.ROLLBACK:
            assert rollback_date is not None , 'rollback_date is required for rollback'
            start , end = CALENDAR.update_schedule(rollback_date , key=key)
            overwrite = True
        elif update_type == UpdateType.RECALC:
            assert start is not None and end is not None , 'start and end are required for recalculate'
            start , end = CALENDAR.update_schedule(max(start , cls.START_DATE) , end , key=key)
            overwrite = True
        else:
            raise ValueError(f'Invalid update type: {update_type}')
        return {'start' : start, 'end' : end, 'overwrite' : overwrite}

    @classmethod
    def proceed_update(cls , start : int | None = None , end : int | None = None , overwrite : bool = False , ref_date : Any | None = None , **kwargs) -> UpdateFlag:
        """proceed the update"""
        raise NotImplementedError(f'proceed_update is not implemented for {cls.__name__}')

    @classmethod
    def update(cls , * , indent : int = 0 , vb_level : Any = 1 , **kwargs) -> UpdateFlag:
        return cls._private_update_method(UpdateType.UPDATE , indent = indent , vb_level = vb_level , **kwargs)

    @classmethod
    def rollback(cls , rollback_date : int , * , indent : int = 0 , vb_level : Any = 1 , **kwargs) -> UpdateFlag:
        return cls._private_update_method(UpdateType.ROLLBACK , rollback_date = rollback_date , indent = indent , vb_level = vb_level , **kwargs)

    @classmethod
    def recalculate(cls , start : int , end : int , * , indent : int = 0 , vb_level : Any = 1 , **kwargs) -> UpdateFlag:
        return cls._private_update_method(UpdateType.RECALC , start = start , end = end , indent = indent , vb_level = vb_level , **kwargs)

    @classmethod
    def _private_update_method(cls , update_type : UpdateType , indent : int = 0 , vb_level : Any = 1 , **kwargs) -> UpdateFlag:
        """parse the update parameters"""
        if not cls._private_handle_update_setup(update_type , indent = indent , vb_level = vb_level , **kwargs):
            return UpdateFlag.SKIPPED
        kwargs = cls._private_handle_update_input(update_type , **kwargs)
        flag = UpdateFlag(cls.proceed_update(indent = indent , vb_level = vb_level , **kwargs))
        cls._private_handle_update_output(flag = flag , **kwargs)
        return flag

    @classmethod
    def _private_handle_update_setup(cls , update_type : UpdateType , indent : int = 0 , vb_level : Any = 1 , **kwargs) -> bool:
        cls.SetClassVB(vb_level , indent)
        if update_type not in cls.ACCEPTABLE_UPDATE_TYPES:
            cls.logger.alert1(f'{update_type.title()} is not supported for {cls.__name__}')
            return False
        return True

    @classmethod
    def _private_handle_update_input(
        cls , update_type : UpdateType , rollback_date : int | None = None , start : int | None = None , end : int | None = None , **kwargs
    ) -> dict[str , Any]:
        
        kwargs = kwargs | {'update_type' : update_type , 'rollback_date' : rollback_date , 'start' : start , 'end' : end}
        kwargs = kwargs | cls.parse_update_input(**kwargs)

        # assert the input parameters
        if update_type == UpdateType.UPDATE:
            cls.logger.note(f'Update since last update!')
        elif update_type == UpdateType.ROLLBACK:
            from src.proj.cal import CALENDAR
            rollback_date = kwargs.get('rollback_date')
            assert rollback_date is not None , 'rollback_date is required for rollback'
            CALENDAR.check_rollback_date(rollback_date)
            cls.logger.note(f'Update since last update!')
        elif update_type == UpdateType.RECALC:
            start , end = kwargs.get('start') , kwargs.get('end')
            assert start is not None and end is not None , \
                ('start and end are required for recalculate , ' 
                 'if you want to recalculate without specifying the range, '
                 'set start and end in parse_update_input to something like 20001001 , 20991231')
            cls.logger.note(f'Recalculate All from {start} to {end}!')
        return kwargs

    @classmethod
    def _private_handle_update_output(
        cls , flag : UpdateFlag , * , update_type : UpdateType , 
        start : int | None = None , end : int | None = None , rollback_date : int | None = None , ref_date : Any | None = None , **kwargs
    ):
        prefix = f'{cls.__name__} {update_type.value.title()}'
        if ref_date:
            flag = flag.with_ref_date(ref_date)
        if flag.ref_date:
            prefix += f' at {flag.ref_date_str}'
        else:
            if update_type == UpdateType.UPDATE:
                from src.proj.cal import CALENDAR
                prefix += f' at {end or CALENDAR.updated()}'
            elif update_type == UpdateType.ROLLBACK:
                prefix += f' from {rollback_date}'
            elif update_type == UpdateType.RECALC:
                prefix += f' from {start} to {end}'
            else:
                raise ValueError(f'Invalid update type: {update_type}')
        
        if flag == UpdateFlag.SKIPPED:
            cls.logger.skipping(f'{prefix} already updated' , idt = 1)
        elif flag == UpdateFlag.SUCCESS:
            cls.logger.success(f'{prefix} updated successfully' , idt = 1 , vb = 1)
        else:
            cls.logger.alert1(f'{prefix} update failed' , idt = 1)