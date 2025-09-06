import warnings
import traceback

class WarningCatcher:
    '''catch specific warnings and show call stack'''
    def __init__(self , catch_warnings : list[str] | None = None):
        self.warnings_caught = []
        self.original_showwarning = warnings.showwarning
        warnings.filterwarnings('always')
        self.catch_warnings = catch_warnings or []
    
    def custom_showwarning(self, message, category, filename, lineno, file=None, line=None):
        # only catch the warnings we care about
        if any(c in str(message) for c in self.catch_warnings):
            stack = traceback.extract_stack()
            print(f"\n caught warning: {message}")
            print(f"warning location: {filename}:{lineno}")
            print("call stack:")
            for i, frame in enumerate(stack[:-1]):  # exclude current frame
                print(f"  {i+1}. {frame.filename}:{frame.lineno} in {frame.name}")
                print(f"     {frame.line}")
            print("-" * 80)
            
            raise Exception(message)
        
        # call original warning show function
        self.original_showwarning(message, category, filename, lineno, file, line)
    
    def __enter__(self):
        warnings.showwarning = self.custom_showwarning
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self.original_showwarning
