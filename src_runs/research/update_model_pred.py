import sys , pathlib
if (path := str(pathlib.Path(__file__).parent.parent.parent)) not in sys.path:
    sys.path.append(path)

from src.api import DataAPI , ModelTrainer , ModelPredictor , ModelHiddenExtractor

if __name__ == '__main__':
    DataAPI.reconstruct_train_data()
    ModelTrainer.update_models()
    ModelHiddenExtractor.update_hidden()
    ModelPredictor.update_factors()