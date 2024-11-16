import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import DataAPI , ModelAPI

if __name__ == '__main__':
    DataAPI.prepare_predict_data()
    ModelAPI.update_hidden()
