import sys

from src.basic import path as PATH

class DualPrinter:
    '''change print target to both terminal and file'''
    def __init__(self, filename : str | None = None):
        self.set_attrs(filename)

    def initiate(self):
        if self.filename is None: return
        self.filename = PATH.log_update.joinpath(self.filename)
        self.filename.parent.mkdir(exist_ok=True,parents=True)
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")

    def set_attrs(self , filename : str | None = None):
        self.filename = filename
        self.initiate()
        return self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass

    def __enter__(self):
        assert self.filename is not None , 'filename is not set'
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.log.close()

    def contents(self):
        assert self.filename is not None , 'filename is not set'
        with open(self.filename , 'r') as f:
            return f.read()