from src.api import Trainer  
app = Trainer.initialize(stage = 0 , resume = 1 , checkname= 1)
app.go()