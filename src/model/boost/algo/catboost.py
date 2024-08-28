import json , tempfile , torch , catboost

from copy import deepcopy
from pathlib import Path
from typing import Any

from .basic import BasicBoosterModel
from ..io import BoosterInput

PLOT_PATH : Path | None = None

class CatBoost(BasicBoosterModel):
    DEFAULT_TRAIN_PARAM = {
        'objective': 'RMSE', # NDCG , RMSE
        'num_boost_round' : 200 , 
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

    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silence = False):
        train_set = self.to_pool(self.data['train'] if train is None else train)
        valid_set = self.to_pool(self.data['valid'] if valid is None else valid)

        train_param = deepcopy(self.train_param)
        train_param = {k:v for k,v in train_param.items() if k in self.DEFAULT_TRAIN_PARAM}
        train_param.update({'random_seed':self.seed , 
                            'task_type': 'GPU' if self.cuda and torch.cuda.is_available() else 'CPU' ,
                            'monotone_constraints' : self.mono_constr(train_param['monotone_constraints'] , train_set.num_col()) ,
                            'verbosity': 0 if silence else 1}) 
        if 'eval_metric' in train_param and train_param['eval_metric'] is None: del train_param['eval_metric']
        num_boost_round = train_param.pop('num_boost_round')
        early_stopping  = train_param.pop('early_stopping')
        verbose_eval    = train_param.pop('verbosity') > 0
        self.cur_train_param = train_param
        
        self.evals_result = dict()
        self.model : catboost.CatBoost = catboost.train(
            train_set, train_param,
            num_boost_round=num_boost_round, 
            early_stopping_rounds = early_stopping,
            evals=[valid_set] , 
            verbose_eval = verbose_eval)

        return self
        
    def predict(self , test : BoosterInput | Any = None):
        test = self.data['test'] if test is None else test
        return test.output(self.model.predict(test.X().cpu().numpy()))
    
    def to_pool(self , input : BoosterInput):
        dset = input.XYW().as_numpy()
        return catboost.Pool(dset.x , dset.y , weight = dset.w)
    
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

    @staticmethod
    def mono_constr(raw_mono_constr , nfeat : int):
        if isinstance(raw_mono_constr , list):
            if len(raw_mono_constr) == 0: 
                mono_constr = None
            elif len(raw_mono_constr) == 1: 
                mono_constr = raw_mono_constr * nfeat
            else: 
                assert len(raw_mono_constr) == nfeat , (len(raw_mono_constr) , nfeat)
                mono_constr = raw_mono_constr
        else:
            mono_constr = [raw_mono_constr] * nfeat
        return mono_constr