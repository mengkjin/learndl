import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import ModelAPI  

if __name__ == '__main__':
    trainer = ModelAPI.initialize_trainer(stage = 0 , resume = 0 , checkname= 1)
    trainer.go()
