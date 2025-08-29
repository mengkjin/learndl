import sys
from io import StringIO

class OutputCapturer:
    def __init__(self):
        self.stdout_capture = StringIO()
        self.stderr_capture = StringIO()
        self.original_stdout = None
        self.original_stderr = None
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def get_stdout(self):
        return self.stdout_capture.getvalue()
    
    def get_stderr(self):
        return self.stderr_capture.getvalue()
    
    def get_output(self):
        return {
            'stdout': self.get_stdout(),
            'stderr': self.get_stderr()
        }
    
    def clear(self):
        self.stdout_capture.seek(0)
        self.stdout_capture.truncate(0)
        
        self.stderr_capture.seek(0)
        self.stderr_capture.truncate(0)