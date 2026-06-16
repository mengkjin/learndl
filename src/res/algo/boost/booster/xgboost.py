"""XGBoost booster wrapper.

Classes:
    XgBoost — :class:`BasicBoostModel` sub-class wrapping ``xgboost.train``.
              Serialisation round-trips through a temporary JSON file because
              ``xgboost.Booster`` has no native ``to_string`` API.
"""
from __future__ import annotations
import json , tempfile , xgboost
from pathlib import Path
from typing import Any

from ..util import BasicBoostModel , BoostInput

__all__ = ['XgBoost']

PLOT_PATH : Path | None = None

class XgBoost(BasicBoostModel):
    """XGBoost wrapper conforming to the :class:`BasicBoostModel` interface.

    Serialisation:
        :meth:`to_dict` / :meth:`load_dict` use :meth:`boost_to_dict` /
        :meth:`boost_from_dict` which save/load via a temporary JSON file to
        work around the lack of a ``model_to_string`` equivalent in XGBoost.
    """
    DEFAULT_TRAIN_PARAM = {
        'booster' : 'gbtree' , # 'dart' , 'gbtree' , 
        'objective': 'mse', # 'mae' , 'rank' , will be converted to 'reg:squarederror' , 'reg:absoluteerror' , 'rank:ndcg'
        'num_boost_round' : 1000 , 
        'early_stopping' : 50 , 
        'rank_target_size' : 100 ,
        'n_bins' : None ,
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
        'lambdarank_pair_method' : 'topk',
        'lambdarank_num_pairs_target' : 100,
        'ndcg_exp_gain' : False,
        'device': 'cpu',
        'seed': 42,
    }
    
    def assert_param(self , **kwargs):
        super().assert_param(**kwargs)
        if self.train_param['objective'] == 'rank':
            self.train_param['objective'] = 'rank:ndcg'
            self.train_param['eval_metric'] = f'ndcg@{self.get_param('rank_target_size' , 100)}'
            self.train_param['lambdarank_pair_method'] = 'topk'
            self.train_param['lambdarank_num_pairs_target'] = self.get_param('rank_target_size' , 100)
        else:
            self.train_param['objective'] = {
                'mse': 'reg:squarederror',
                'mae': 'reg:absoluteerror',
            }[self.train_param['objective']]
            self.train_param['eval_metric'] = None
            self.train_param['lambdarank_pair_method'] = None
            self.train_param['lambdarank_num_pairs_target'] = None
        return self

    def fit(self , train : BoostInput | Any = None , valid : BoostInput | Any = None , silent = False):
        self.boost_fit_inputs(train , valid , silent)

        # group: cross-section queries by date (required for rank:* objectives e.g. rank:ndcg).
        train_set = self.fit_train_ds.to_xgboost_dataset(rank=self.is_rankor)
        valid_set = self.fit_valid_ds.to_xgboost_dataset(rank=self.is_rankor)
        self.fit_train_param.update({
            'seed':                 self.seed , 
            'device':               'gpu' if self.use_gpu else 'cpu' , 
            'verbosity':            0 if silent else 1 ,
            'monotone_constraints': self.mono_constr(self.fit_train_param , self.fit_train_ds.nfeat , as_tuple=True)}) 

        num_boost_round = self.fit_train_param.pop('num_boost_round')
        early_stopping  = self.fit_train_param.pop('early_stopping')
        verbose_eval    = self.fit_train_param['verbosity']

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
        
    def predict(self , x : BoostInput | Any = 'test'):
        data = self.boost_input(x)
        X = data.Dataset().to_xgboost_dataset(rank=self.is_rankor)
        return data.output(self.model.predict(X))
    
    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['model'] = self.boost_to_dict(self.model)
        return model_dict
    
    def load_dict(self , model_dict : dict , cuda = False , seed = None):
        super().load_dict(model_dict , cuda , seed)
        self.model = self.boost_from_dict(model_dict['model'])
        return self

    @staticmethod
    def boost_to_dict(model : xgboost.Booster):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir).joinpath('model.json') 
            model.save_model(model_path)
            with open(model_path , encoding='utf-8') as file: 
                model_dict = json.load(file)  
        return model_dict
    
    @staticmethod
    def boost_from_dict(model_dict : dict):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = Path(tempdir).joinpath('model.json')
            with open(model_path, 'w' , encoding='utf-8') as file: 
                json.dump(model_dict , file, indent=4)
            model = xgboost.Booster(model_file=model_path)
        return model