"""catch specific warnings and show call stack"""
from __future__ import annotations
import warnings

from typing import Any
from src.proj.bases import BoundLogger

__all__ = ['WarningCatcher']

class WarningCatcher(BoundLogger):
    """
    catch specific warnings and show call stack
    example:
        with WarningCatcher(['This will raise an exception']):
            raise Exception('This will raise an exception')
    """
    def __init__(
        self , 
        raise_warnings : list[str] | None = None , 
        ignore_warnings : list[str] | None = None , * ,
        highlight_varibles : dict[str, Any] | None = None ,
        indent: int = 0 , vb_level: int = 1 , **kwargs
    ):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.original_showwarning = warnings.showwarning
        warnings.filterwarnings('always')
        self.raise_warnings = [] if raise_warnings is None else [c.lower() for c in raise_warnings]
        self.ignore_warnings = [] if ignore_warnings is None else [c.lower() for c in ignore_warnings]
        self.highlight_varibles = highlight_varibles
    
    def custom_showwarning(self, message, category, filename, lineno, file=None, line=None) -> None:
        """Custom warning show function to catch specific warnings and show call stack"""
        # only catch the warnings we care about
        if any(c in str(message).lower() for c in self.raise_warnings):
            self.logger.alert1(f"\n caught warning: {message}")
            self.logger.alert1(f"warning location: {filename}:{lineno}")
            self.logger.alert1("call stack:")
            self.logger.print_traceback_stack()
            self.logger.alert1("-" * 80)

            if self.highlight_varibles is not None:
                for var_name, var_value in self.highlight_varibles.items():
                    self.logger.alert1(f"{var_name}: {var_value}")
                self.logger.alert1("-" * 80)
                
            raise Exception(message)

        if any(c in str(message).lower() for c in self.ignore_warnings):
            return
        
        # call original warning show function
        self.original_showwarning(message, category, filename, lineno, file, line)
    
    def __enter__(self):
        warnings.showwarning = self.custom_showwarning
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self.original_showwarning