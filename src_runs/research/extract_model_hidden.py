import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import ModelHiddenExtractor

if __name__ == '__main__':
    ModelHiddenExtractor.update_hidden()