import json , tempfile , xgboost

from pathlib import Path
from typing import Any

from ..util import BasicBoosterModel , BoosterInput

PLOT_PATH : Path | None = None

class XgBoost(BasicBoosterModel):
    DEFAULT_TRAIN_PARAM = {
        'booster' : 'gbtree' , # 'dart' , 'gbtree' , 
        'objective': 'reg:squarederror', # 'reg:squarederror', 'reg:absoluteerror' , multi:softmax
        'num_boost_round' : 100 , 
        'early_stopping' : 50 , 
        'eval_metric' : None , # rank:ndcg ,
        'subsample': 1.,
        'colsample_bytree':1.,
        'verbosity': 1 , 
        'learning_rate': 0.3, 
        'reg_lambda': 1e-05, 
        'reg_alpha': 1e-07, 
        'max_depth': 6, 
        'monotone_constraints': 0 , 
        'rate_drop' : 0.1,
        'device': 'cpu',
        'seed': 42,
    }

    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silent = False):
        self.booster_fit_inputs(train , valid , silent)

        train_set = xgboost.DMatrix(**self.fit_train_ds.booster_inputs('xgboost'))
        valid_set = xgboost.DMatrix(**self.fit_valid_ds.booster_inputs('xgboost'))
        self.fit_train_param.update({
            'seed':                 self.seed , 
            'device':               'gpu' if self.use_gpu else 'cpu' , 
            'verbosity':            0 if silent else 1 ,
            'monotone_constraints': self.mono_constr(self.fit_train_param , self.fit_train_ds.nfeat , as_tuple=True)}) 

        num_boost_round = self.fit_train_param.pop('num_boost_round')
        early_stopping  = self.fit_train_param.pop('early_stopping')
        verbose_eval    = self.fit_train_param['verbosity']
        num_class       = self.fit_train_param.pop('n_bins' , None)
        if self.fit_train_param['objective'] in ['softmax']: 
            self.fit_train_param['num_class'] = num_class

        self.evals_result = dict()
        self.model : xgboost.Booster = xgboost.train(
            dtrain = train_set,
            params = self.fit_train_param, 
            num_boost_round=num_boost_round, 
            early_stopping_rounds = early_stopping,
            evals=[(valid_set, 'eval')] , 
            evals_result=self.evals_result , 
            verbose_eval = verbose_eval)

        return self
        
    def predict(self , x : BoosterInput | str = 'test'):
        data = self.booster_input(x)
        X = xgboost.DMatrix(**data.Dataset().booster_inputs('xgboost'))
        return data.output(self.model.predict(X))
    
    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['model'] = self.booster_to_dict(self.model)
        return model_dict
    
    def load_dict(self , model_dict : dict , cuda = False , seed = None):
        super().load_dict(model_dict , cuda , seed)
        self.model = self.booster_from_dict(model_dict['model'])
        return self

    @staticmethod
    def booster_to_dict(model : xgboost.Booster):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir).joinpath('model.json') 
            model.save_model(model_path)
            with open(model_path, 'r') as file: 
                model_dict = json.load(file)  
        return model_dict
    
    @staticmethod
    def booster_from_dict(model_dict : dict):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir).joinpath('model.json')
            with open(model_path, 'w') as file: 
                json.dump(model_dict , file, indent=4)
            model = xgboost.Booster(model_file=model_path)
        return model