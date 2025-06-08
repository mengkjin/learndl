from src.api import ModelAPI  
app = ModelAPI.Trainer.initialize(stage = 0 , resume = 0 , checkname= 1 , 
                                  module= None , short_test=True)
app.go()