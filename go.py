from src.api import ModelTrainer  
app = ModelTrainer.initialize(stage = 0 , resume = 0 , checkname= 1)
app.go()