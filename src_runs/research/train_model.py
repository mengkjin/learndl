import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import ModelTrainer  

if __name__ == '__main__':
    app = ModelTrainer.initialize(stage = 0 , resume = 0 , checkname= 1)
    app.go()