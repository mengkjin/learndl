from src2.api import Trainer  
app = Trainer.initialize(stage = 0 , resume = 0 , checkname= 1)
app.go()