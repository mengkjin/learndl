import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import DataAPI , ModelAPI

if __name__ == '__main__':
    DataAPI.reconstruct_train_data()
    ModelAPI.update_models()
    ModelAPI.update_hidden()
    ModelAPI.update_factors()
