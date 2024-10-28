import json , tempfile , torch , catboost

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..util.basic import BasicBoosterModel
from ..util.io import BoosterInput

PLOT_PATH : Path | None = None

class CatBoost(BasicBoosterModel):
    DEFAULT_TRAIN_PARAM = {
        'objective': 'RMSE', # NDCG , RMSE
        'num_boost_round' : 100 , 
        'early_stopping' : 50 , 
        'verbosity': 1 , 
        'learning_rate': 0.3, 
        'l2_leaf_reg': 1e-05, 
        'max_depth': 6, 
        'monotone_constraints': 0 , 
        'random_strength' : 1.,
        'task_type': 'CPU',
        'random_seed': 42,
        'allow_writing_files' : False ,
    }

    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silent = False):
        self.booster_fit_inputs(train , valid , silent)

        train_set = catboost.Pool(**self.fit_train_ds.catboost_inputs())
        valid_set = catboost.Pool(**self.fit_valid_ds.catboost_inputs())

        num_boost_round = self.fit_train_param.pop('num_boost_round')
        early_stopping  = self.fit_train_param.pop('early_stopping')
        verbose_eval    = self.fit_train_param.pop('verbosity') > 0
        num_class       = self.fit_train_param.pop('n_bins' , None)
        if 'eval_metric' in self.fit_train_param and self.fit_train_param['eval_metric'] is None: del self.fit_train_param['eval_metric']

        self.fit_train_param.update({
            'random_seed':          self.seed , 
            'task_type':            'GPU' if self.use_gpu else 'CPU' , 
            'verbosity':            0 if silent else 1 ,
            'monotone_constraints': self.mono_constr(self.fit_train_param , self.fit_train_ds.nfeat , as_tuple=True)}) 
        if self.fit_train_param['objective'] in ['softmax']: self.fit_train_param['num_class'] = num_class
               
        self.evals_result = dict()
        self.model : catboost.CatBoost = catboost.train(
            dtrain = train_set,
            params = self.fit_train_param,
            num_boost_round=num_boost_round, 
            early_stopping_rounds = early_stopping,
            evals=[valid_set] , 
            verbose_eval = verbose_eval)

        return self
        
    def predict(self , x : BoosterInput | str = 'test'):
        data = self.booster_input(x)
        return data.output(self.model.predict(data.X().cpu().numpy()))
    
    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['model'] = self.booster_to_dict(self.model)
        return model_dict
    
    def load_dict(self , model_dict : dict , cuda = False , seed = None):
        super().load_dict(model_dict , cuda , seed)
        self.model = self.booster_from_dict(model_dict['model'])
        return self

    @staticmethod
    def booster_to_dict(model : catboost.CatBoost):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir).joinpath('model.json') 
            model.save_model(model_path , format='json')
            with open(model_path, 'r') as file: model_dict = json.load(file)  
        return model_dict
    
    @staticmethod
    def booster_from_dict(model_dict : dict):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir).joinpath('model.json')
            with open(model_path, 'w') as file: json.dump(model_dict , file, indent=4)
            model = catboost.CatBoost()
            model.load_model(model_path , format= 'json')
        return model