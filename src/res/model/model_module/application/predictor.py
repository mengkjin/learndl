from __future__ import annotations
import torch
import numpy as np
import pandas as pd
import polars as pl

from functools import cached_property
from typing import Any , ClassVar , Literal , overload

from src.proj import MACHINE , Proj , CALENDAR
from src.proj.core import strPath
from src.proj.util import RequireGrad , BaseModule
from src.res.model.util import PredictorPath , ModelConfig , DataModule
from src.res.model.model_module.module import get_predictor_module

class ArchivedPredictorModel(BaseModule):
    '''for a model to predict recent/history data'''
    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    @overload
    def __init__(self , predictor_path : PredictorPath , / , * , indent : int = 1 , vb_level : Any = 1):
        """Initialize from a PredictorPath object"""
    @overload
    def __init__(self , model_input : strPath | None | Any , 
        model_num : int | list[int] | range | Literal['all'] | Any | None = None ,
        submodel : str = 'best' ,
        pred_name : str | None = None , * , indent : int = 1 , vb_level : Any = 1):
        """Initialize from a model input, and convert to PredictorPath object"""
    def __init__(self , model_input : strPath | None | Any | PredictorPath, 
        model_num : int | list[int] | range | Literal['all'] | Any | None = None ,
        submodel : str | None = 'best' , pred_name : str | None = None , * , 
        indent : int = 1 , vb_level : Any = 1):
        self.set_vb(vb_level , indent)
        if isinstance(model_input , PredictorPath):
            self.path = model_input
        else:
            assert model_num is not None and submodel is not None , 'model_num and submodel must be provided'
            self.path = PredictorPath(model_input , model_num , submodel , pred_name)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(path={self.path})'

    def __call__(self , date : int):
        if self.cached_df.empty or date not in self.cached_df['date'].unique():
            self.predict_dates([date])
        return self.cached_df.query('date == @date')

    @classmethod
    def from_path(cls , path : PredictorPath):
        """Initialize from a PredictorPath object"""
        return cls(path)

    @classmethod
    def from_model_str(cls , model_str : str):
        """
        Initialize from a model string
        the model string is like 'gru@gru_avg@0@best' , where @ separates model_path_input , model_num , model_submodel
        """
        args = model_str.rsplit('@' , 2)
        assert len(args) in [2,3] , f'Invalid model string: {model_str} , {args}'
        if len(args) == 2:
            model_path_input , model_num = args
            submodel = 'best'
        elif args[-1] not in ['best' , 'swalast' , 'swabest'] and args[-1].isdigit():
            model_path_input = '@'.join(args[:-1])
            model_num = args[-1]
            submodel = 'best'
        else:
            model_path_input , model_num , submodel = args
        return cls(model_path_input , int(model_num) , submodel)

    @property
    def model_name(self) -> str:
        return self.path.model_name

    @property
    def pred_name(self) -> str:
        return self.path.pred_name

    @cached_property
    def cached_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    @cached_property
    def current_update_dates(self) -> list[Any]:
        return []

    @cached_property
    def current_deploy_dates(self) -> list[Any]:
        return []

    @cached_property
    def config(self) -> ModelConfig:
        with Proj.silence:
            return ModelConfig(self.model_name , stage=2 , resume=1).start_model()

    @cached_property
    def model(self):
        return get_predictor_module(self.config)

    @cached_property
    def model_dates(self) -> np.ndarray:
        return self.path.model_dates

    @cached_property
    def model_nums(self) -> np.ndarray:
        return self.path.model_nums

    @cached_property
    def model_submodels(self) -> np.ndarray:
        return self.path.model_submodels

    def load_data(self , min_date : int | None = None , max_date : int | None = None):
        updated = CALENDAR.updated()
        min_date = min_date or 20170101
        max_date = max_date or min_date + 20000 # 2 years
        if min_date > CALENDAR.today(-100):
            use_data = 'predict'
        elif max_date < updated:
            use_data = 'fit'
        else:
            use_data = 'both'

        if not hasattr(self , 'data_module'):
            self.data = DataModule(self.config , use_data).load_data() 
        elif self.data.use_data != 'both' and self.data.use_data != use_data:
            self.data = DataModule(self.config , 'both').load_data() 
        return self
    
    def update_preds(self , update = True , overwrite = False , start = None , end = None):
        '''get update dates and predict these dates'''
        assert update != overwrite , 'update and overwrite must be different here'
        
        dates = CALENDAR.slice(CALENDAR.diffs(self.path.pred_target_dates , self.path.pred_dates if update else []) , start , end)
        with Proj.silence:
            self.predict_dates(dates)
        self.save_preds()
        self.deploy()

    def _get_model_num_and_submodel(self , model_num : int | None = None , submodel : str | None = None):
        if model_num is None:
            assert len(self.path.use_model_nums) == 1 , f'model_num must be provided when there are multiple model numbers'
            model_num = int(self.path.use_model_nums[0])
        else:
            assert model_num in self.model_nums , f'model_num {model_num} not in {self.model_nums}'
        if submodel is None:
            submodel = self.path.use_submodel
        else:
            assert submodel in self.model_submodels , f'submodel {submodel} not in {self.model_submodels}'
        return model_num , submodel

    def batch_data(
        self , date : int , model_date : int | None = None , * , 
        model_num : int | None = None , submodel : str | None = None , 
        retro_start_date : int | None = None , retro_end_date : int | None = None ,
        require_grad = False , silent = True
    ):
        """
        calculate the batch data of a given date
        Args:
            date : int, the query date
            model_date : int | None = None, the model date of the model archive , None means the closest model date before the query date (or the first model date)
            model_num : int | None = None, the model number of the model archive, None use default model number
            submodel : str | None = None, the model submodel, None use default model submodel
            retro_start_date : int | None = None, if given, init the retrospective_dataloader with this start date
            retro_end_date : int | None = None, if given, init the retrospective_dataloader with this end date
            require_grad : bool = False, whether to require gradient
            silent : bool = True, whether to silence the warning
        Returns:
            BatchData(batch_input , batch_output) of date
        """
        model_num , submodel = self._get_model_num_and_submodel(model_num , submodel)
        if model_date is None:
            prev_model_dates = self.model_dates[self.model_dates < date]
            model_date = prev_model_dates[-1] if len(prev_model_dates) > 0 else self.model_dates[0]
        assert model_date is not None and model_date in self.model_dates , f'model_date {model_date} not in {self.model_dates}'
        with Proj.silence(silent):
            model_param = self.config.model_param[model_num]
            self.load_data(date)
            self.data.setup('retrospective' , model_param , date , retro_start_date = retro_start_date , retro_end_date = retro_end_date)
            model = self.model.load_model(model_num , model_date , self.path.use_submodel , model_param = model_param , cache_model = True)
            self.dataloader = self.data.retrospective_dataloader()
        with RequireGrad(require_grad):
            batch_input = self.dataloader.of_date(date)
            return model.get_batch_data(batch_input)

    def iter_batch_data(
        self , start_date : int , end_date : int , model_date : int , 
        * , model_num : int | None = None , submodel : str | None = None , 
        require_grad = False , silent = True):
        """
        Iterate batch data of a given model number, model date, start date, and end date
        Args:
            model_date : int, the model date of the archived model
            start_date : int, the start date
            end_date : int, the end date
            model_num : int | None = None, the model number of the model archive, None use default model number
            submodel : str | None = None, the model submodel, None use default model submodel
            require_grad : bool = False, whether to require gradient
        Returns:
            Iterator of BatchData(batch_input , batch_output)
        """
        model_num , submodel = self._get_model_num_and_submodel(model_num , submodel)
        assert model_date in self.model_dates , f'model_date {model_date} not in {self.model_dates}'
        assert start_date < end_date , f'start_date {start_date} must be less than end_date {end_date}'
        model_param = self.config.model_param[model_num]
        with Proj.silence(silent):
            self.load_data(start_date , end_date)
            self.data.setup('retrospective' , model_param , start_date , retro_start_date = start_date , retro_end_date = end_date)
            model = self.model.load_model(model_num , model_date , submodel , model_param = model_param , cache_model = True)
            self.dataloader = self.data.retrospective_dataloader()
        with RequireGrad(require_grad):
            for batch_input in self.dataloader:
                yield model.get_batch_data(batch_input)

    def hidden_block(
        self , start_date : int , end_date : int , model_date : int , * , 
        model_num : int | None = None , submodel : str | None = None , feature_prefix : bool = True , silent = True):
        """
        Iterate hidden block of a given model number, model date, start date, and end date
        Args:
            model_date : int, the model date of the archived model
            start_date : int, the start date
            end_date : int, the end date
            model_num : int | None = None, the model number of the model archive, None use default model number
            submodel : str | None = None, the model submodel, None use default model submodel
            feature_prefix : bool = True, whether to add feature prefix to the column names
        Returns:
            DataBlock of hiddens
        """
        model_num , submodel = self._get_model_num_and_submodel(model_num , submodel)
        hidden_dfs : list[pl.DataFrame] = []
        for batch_data in self.iter_batch_data(start_date , end_date , model_date , model_num = model_num , submodel = submodel , require_grad = False , silent = silent):
            if batch_data.output.empty or batch_data.batch_date in self.data.early_test_dates:
                continue
            hidden_dfs.append(batch_data.hidden_df_pl())
        df = pl.concat(hidden_dfs , how = 'vertical_relaxed')
        from src.data import DataBlock
        block = DataBlock.from_polars(df)
        if feature_prefix:
            prefix = '@'.join([self.config.model_module , self.config.model_clean_name , str(model_num) , submodel])
            block.update(feature = [f'{prefix}.{col}' for col in block.feature])
        return block

    def predict_dates(self , dates : np.ndarray | list[int]):
        '''predict recent days'''
        if len(dates) == 0: 
            return self
        dates = np.array(dates)
        self.load_data(dates.min())
        pred_dates = dates[dates <= max(self.data.test_full_dates)]
        if pred_dates.size == 0: 
            return self
        assert any(self.path.model_dates < pred_dates.min()) , f'no model date before {pred_dates}'
        df_task = pd.DataFrame({'pred_dates' : pred_dates , 
                                'model_date' : [max(self.path.model_dates[self.path.model_dates < d]) for d in pred_dates] , 
                                'calculated' : 0})
        torch.set_grad_enabled(False)
        df_list : list[pd.DataFrame] = []
        
        for model_date , df_sub in df_task.query('calculated == 0').groupby('model_date'):
            for model_num in self.path.use_model_nums:
                model_param = self.config.model_param[model_num]
                assert isinstance(model_date , int) , model_date
                self.data.setup('retrospective' ,  model_param , model_date)
                model = self.model.load_model(model_num , model_date , self.path.use_submodel , model_param = model_param)
                
                tdates = self.data.model_test_dates
                within = np.isin(tdates , df_sub.query('calculated == 0')['pred_dates'])
                loader = self.data.retrospective_dataloader()

                for tdate , do_calc , batch_input in zip(tdates , within , loader):
                    if not do_calc or len(batch_input) == 0: 
                        continue
                    df = model.get_batch_data(batch_input).pred_df(colnames = self.model_name , model_num = model_num)
                    df_list.append(df)
                    df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1

        if df_list:
            self.cached_df = pd.concat(df_list , axis = 0).groupby(['date','secid'])[self.model_name].mean().reset_index()
        return self

    def save_preds(self , df : pd.DataFrame | None = None , overwrite = False , secid_col = SECID_COLS , date_col = DATE_COLS):
        if df is None:
            df = self.cached_df
        if df.empty: 
            return self
        for date , subdf in df.groupby(date_col):
            subdf = subdf.drop(columns='date').set_index(secid_col)
            self.path.save_pred(subdf , date , overwrite , indent = 2 , vb_level = 3)
            self.current_update_dates.append(date)
        return self

    @property
    def deploy_required(self) -> bool:
        return MACHINE.hfm_factor_dir is not None

    def deploy(self , overwrite = False):
        '''deploy df by day to class.destination'''
        if MACHINE.hfm_factor_dir is None: 
            return self
        try:
            path_deploy = MACHINE.hfm_factor_dir.joinpath(self.path.pred_name)
            path_deploy.parent.mkdir(parents=True,exist_ok=True)
            if overwrite:
                dates = self.path.pred_dates
            else:
                deployed_dates = [int(path.name.removesuffix('.txt').split('_')[-1]) for path in path_deploy.glob('*.txt')]
                dates = np.setdiff1d(self.path.pred_dates , deployed_dates)

            for date in dates:
                df = self.path.load_pred(date , vb_level = 'never')
                df.to_csv(path_deploy.joinpath(f'{self.path.pred_name}_{date}.txt') , sep='\t', index=False, header=False)
                self.current_deploy_dates.append(date)
        except OSError as e:
            self.logger.error(f'{self.path.pred_name} deploy error: {e}')

        return self
    
    def df_corr(self , df = None , window = 30 , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''prediction correlation of ecent days'''
        if df is None: 
            df = self.cached_df
        if df is None: 
            return NotImplemented
        dates : Any = df[date_col].unique()
        dates = np.sort(dates)[-window:]
        df = df.query(f'{date_col} in @dates')
        assert isinstance(df , pd.DataFrame) , f'{type(df)} is not a DataFrame'
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()

    @classmethod
    def get_model(cls , model_name : str):
        model = PredictorPath.SelectModels(model_name)[0]
        return cls(model)

    @classmethod
    def update(cls , model_name : str | None = None , start = None , end = None , indent : int = 0 , vb_level : Any = 1):
        '''Update prediction factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        cls.logger.note('Update since last update!' , ind = indent , vb = vb_level)
        if start is not None or end is not None:
            cls.logger.stdout(f'Update from {start} to {end}' , ind = indent + 1 , vb = vb_level)
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , ind = indent + 1 , vb = vb_level)
        for model in models:
            md = cls(model , indent = indent + 1 , vb_level = vb_level + 1)
            md.update_preds(update = True , overwrite = False , start = start , end = end)
            if md.current_update_dates:
                md.logger.success(f'Update model prediction for {model} , len={len(md.current_update_dates)}')
            else:
                md.logger.skipping(f'Model prediction for {model} is up to date')
            if md.deploy_required:
                if md.current_deploy_dates:
                    md.logger.success(f'Deploy model prediction for {model} , len={len(md.current_deploy_dates)}')
                else:
                    md.logger.skipping(f'Model prediction for {model} is up to date')
        return md

    @classmethod
    def recalculate(cls , model_name : str | None = None , start = None , end = None , indent : int = 0 , vb_level : Any = 1):
        """Recalculate all model predictions"""
        cls.logger.note('Recalculate All!' , ind = indent , vb = vb_level)
        if start is not None or end is not None:
            cls.logger.stdout(f'Recalculate from {start} to {end}' , ind = indent + 1 , vb = vb_level)
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , ind = indent + 1 , vb = vb_level)
        for model in models:
            md = cls(model , indent = indent + 1 , vb_level = vb_level + 1)
            md.update_preds(update = False , overwrite = True , start = start , end = end)
            if md.current_update_dates:
                md.logger.success(f'Finish recalculating model prediction for {model} , len={len(md.current_update_dates)}')
            else:
                md.logger.skipping(f'No new recalculating model prediction for {model}')
            if md.deploy_required:
                if md.current_deploy_dates:
                    md.logger.success(f'Finish deploying model prediction for {model} , len={len(md.current_deploy_dates)}')
                else:
                    md.logger.skipping(f'No new deploying model prediction for {model}')
        return md
