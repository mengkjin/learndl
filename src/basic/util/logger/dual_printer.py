import sys

from src.basic import path as PATH

class DualPrinter:
    '''change print target to both terminal and file'''
    def __init__(self, filename : str):
        self.filename = PATH.log_update.joinpath(filename)
        self.filename.parent.mkdir(exist_ok=True,parents=True)
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.log.close()

    def contents(self):
        with open(self.filename , 'r') as f:
            return f.read()