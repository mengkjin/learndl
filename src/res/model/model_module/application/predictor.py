from __future__ import annotations
import torch
import numpy as np
import pandas as pd
import polars as pl

from functools import cached_property
from typing import Any , ClassVar , Literal , overload

from src.proj import MACHINE , DB , Proj , CALENDAR , BaseClass , BaseType
from src.proj.util import RequireGrad , AsyncSaver
from src.data.util import DataBlock
from src.res.model.util import PredictorPath , ModelConfig , DataModule
from src.res.model.model_module.module import get_predictor_module

class ArchivedPredictorModel(BaseClass.BoundLogger):
    '''for a model to predict recent/history data'''
    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    @overload
    def __init__(self , model_input : PredictorPath , / , * , indent : int = 0 , vb_level : Any = 1):
        """Initialize from a PredictorPath object"""
    @overload
    def __init__(self , model_input : BaseType.strPath | None | Any , 
        model_num : int | list[int] | range | Literal['all'] | Any | None = None ,
        submodel : str = 'best' , pred_name : str | None = None , / , 
        indent : int = 0 , vb_level : Any = 1):
        """Initialize from a model input, and convert to PredictorPath object"""
    def __init__(self , model_input : BaseType.strPath | None | Any | PredictorPath, 
        model_num : int | list[int] | range | Literal['all'] | Any | None = None ,
        submodel : str | None = 'best' , pred_name : str | None = None , / , 
        indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
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
        if self.current_update_dates:
            self.logger.success(f'Update model prediction for {self.pred_name} , len={len(self.current_update_dates)}')
        else:
            self.logger.skipping(f'Model prediction for {self.pred_name} is up to date')
        if self.deploy_required and self.current_deploy_dates:
            self.logger.success(f'Deploy model prediction for {self.pred_name} , len={len(self.current_deploy_dates)}')
        elif self.deploy_required:
            self.logger.skipping(f'Model prediction for {self.pred_name} is up to date')

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

    def _get_dates(self , dates : np.ndarray | list[int] , start : int | None = None , end : int | None = None , step : int = 1):
        if start is not None or end is not None:
            assert start is not None and end is not None , 'start and end must be provided together'
            assert start < end , f'start {start} must be less than end {end}'
            assert step > 0 , f'step {step} must be greater than 0'
            dates = CALENDAR.range(start , end , step = step)
        return np.array(dates)

    def iter_batch_data(
        self , dates : np.ndarray | list[int] , model_date : int , 
        * , start : int | None = None , end : int | None = None , step : int = 1 , model_num : int | None = None , submodel : str | None = None , 
        require_grad = False , silent = True):
        """
        Iterate batch data of a given model number, model date, start date, and end date
        Args:
            dates : np.ndarray | list[int], the dates to iterate
            model_date : int, the model date of the archived model
            start : int | None = None, the start date, if given, iterate dates from start to end with step
            end : int | None = None, the end date, if given, iterate dates from start to end with step
            step : int = 1, the step size
            model_num : int | None = None, the model number of the model archive, None use default model number
            submodel : str | None = None, the model submodel, None use default model submodel
            require_grad : bool = False, whether to require gradient
        Returns:
            Iterator of BatchData(batch_input , batch_output)
        """
        model_num , submodel = self._get_model_num_and_submodel(model_num , submodel)
        assert model_date in self.model_dates , f'model_date {model_date} not in {self.model_dates}'
        dates = self._get_dates(dates , start , end , step)
        if len(dates) == 0:
            return iter([])
        start_date , end_date = min(dates) , max(dates)
        model_param = self.config.model_param[model_num]
        with Proj.silence(silent):
            self.load_data(start_date , end_date)
            self.data.setup('retrospective' , model_param , start_date , retro_start_date = start_date , retro_end_date = end_date)
            model = self.model.load_model(model_num , model_date , submodel , model_param = model_param , cache_model = True)
            self.dataloader = self.data.retrospective_dataloader()
        with RequireGrad(require_grad):
            for batch_input in self.dataloader:
                batch_data = model.get_batch_data(batch_input)
                if len(self.data.early_test_dates) == 0 and batch_input.date0 not in dates:
                    # if no early test dates, and the batch date is not in dates, no need to warmup,skip
                    continue
                batch_data = model.get_batch_data(batch_input)
                if batch_data.batch_date in dates:
                    yield batch_data

    def hidden_block(
        self , 
        dates : np.ndarray | list[int] , model_date : int , * , 
        start = None , end = None , step : int = 1 ,
        model_num : int | None = None , submodel : str | None = None , feature_prefix : bool = True , 
        align_secid : np.ndarray | None = None , align_date : np.ndarray | None = None ,
        load_first = True , silent = True
    ) -> DataBlock:
        """
        Iterate hidden block of a given model number, model date, start date, and end date
        Args:
            dates : np.ndarray | list[int], the dates to iterate
            model_date : int, the model date of the archived model
            start : int | None = None, the start date, if given, iterate dates from start to end with step
            end : int | None = None, the end date, if given, iterate dates from start to end with step
            step : int = 1, the step size
            model_num : int | None = None, the model number of the model archive, None use default model number
            submodel : str | None = None, the model submodel, None use default model submodel
            feature_prefix : bool = True, whether to add feature prefix to the column names
            load_first : bool = True, whether to load the first batch data to get the hidden block
            silent : bool = True, whether to silence the warning
        Returns:
            DataBlock of hiddens
        """
        model_num , submodel = self._get_model_num_and_submodel(model_num , submodel)
        dates = self._get_dates(dates , start , end , step)
        if len(dates) == 0:
            return DataBlock()

        hidden_path = self.hidden_values_path(model_num , model_date , submodel)
        hidden_dfs : list[pl.DataFrame] = []
        existing_dates = []
        saved_hidden_df = DB.load_df_pl(hidden_path)
        if not load_first and saved_hidden_df.height > 0:
            saved_hidden_df = saved_hidden_df.filter(~pl.col('date').is_in(dates))
        if saved_hidden_df.height > 0:
            hidden_dfs.append(saved_hidden_df)
        existing_dates = saved_hidden_df['date'].unique()
        dates = np.setdiff1d(dates , existing_dates)
            
        batch_data_iterator = self.iter_batch_data(
            dates , model_date , model_num = model_num , 
            submodel = submodel , require_grad = False , silent = silent)
        for batch_data in batch_data_iterator:
            if batch_data.output.empty or batch_data.batch_date in self.data.early_test_dates:
                continue
            assert batch_data.batch_date not in existing_dates , f'batch_data.batch_date {batch_data.batch_date} already in {existing_dates}'
            hidden_dfs.append(batch_data.hidden_df_pl())
        df = pl.concat(hidden_dfs , how = 'vertical_relaxed')
        if df.height > 0 and len(dates) > 0:
            AsyncSaver.df(
                df , hidden_path , overwrite = True , 
                prefix = f'{self.pred_name} Hidden Values' , indent = self.logger.indent + 1 , vb_level = self.logger.vb_level + 1)
            
        if align_secid is not None:
            df = df.filter(pl.col('secid').is_in(align_secid))
        if align_date is not None:
            df = df.filter(pl.col('date').is_in(align_date))

        block = DataBlock.from_polars(df)
        if feature_prefix:
            prefix = '@'.join([self.config.model_module , self.config.model_clean_name , str(model_num) , submodel])
            block.update(feature = [f'{prefix}.{col}' for col in block.feature])

        block = block.align_secid_date(align_secid , align_date)
        return block

    def hidden_values_path(self , model_num : int , model_date : int , submodel : str):
        return self.path.snapshot('hidden_values' , f'{model_num}.{model_date}.{submodel}.feather')

    def predict_dates(self , dates : np.ndarray | list[int]):
        '''predict recent days'''
        if len(dates) == 0: 
            return self
        dates = np.array(dates)
        self.load_data(dates.min() , dates.max())
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
            assert isinstance(model_date , int) and model_date in self.model_dates , \
                f'model_date {model_date} not in {self.model_dates}'
            for model_num in self.path.use_model_nums:
                model_param = self.config.model_param[model_num]
                tdates = df_sub['pred_dates'].unique()
                self.data.setup('retrospective' ,  model_param , model_date , retro_start_date = tdates.min() , retro_end_date = tdates.max())
                model = self.model.load_model(model_num , model_date , self.path.use_submodel , model_param = model_param)
                
                for batch_input in self.data.retrospective_dataloader():
                    tdate = batch_input.date0
                    if tdate not in pred_dates or len(batch_input) == 0: 
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
            self.path.set_vb(self.vb_level + 1 , self.indent + 1)
            self.path.save_pred(subdf , date , overwrite)
            self.current_update_dates.append(date)
        if not self.current_update_dates:
            self.logger.stdout(df)
            raise ValueError(f'No update dates for {self.path.pred_name}')
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
        cls.SetClassVB(vb_level , indent)
        cls.logger.note('Update since last update!')
        if start is not None or end is not None:
            cls.logger.stdout(f'Update from {start} to {end}' , idt = 1)
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , idt = 1)
        for model in models:
            md = cls(model , indent = indent , vb_level = vb_level)
            with md.logger.subprocess(idt = 1 , vb = 1):
                md.update_preds(update = True , overwrite = False , start = start , end = end)
        return md

    @classmethod
    def recalculate(cls , model_name : str | None = None , start = None , end = None , indent : int = 0 , vb_level : Any = 1):
        """Recalculate all model predictions"""
        cls.SetClassVB(vb_level , indent)
        cls.logger.note('Recalculate All!')
        if start is not None or end is not None:
            cls.logger.stdout(f'Recalculate from {start} to {end}' , idt = 1)
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , idt = 1)
        for model in models:
            md = cls(model)
            with md.logger.subprocess(idt = 1 , vb = 1):
                md.update_preds(update = False , overwrite = True , start = start , end = end)
        return md
