from __future__ import annotations

import numpy as np
import pandas as pd
from functools import cached_property
from pathlib import Path

from src.proj import DB , PATH , Const , CALENDAR
from src.proj.util import AsyncSaver
from src.res.model.util.config import ModelConfig
from .pipeline import TrainerPipeline
from .base_trainer import BaseTrainer

class PredRecorder(TrainerPipeline):
    """Trainer predictor recorder class, used to record the predictor results"""
    PRED_KEYS = ['model_num' , 'model_date' , 'submodel' , 'batch_idx']
    PRED_IDXS = ['secid' , 'date']
    PRED_COLS = ['pred' , 'label']
    def __init__(self , trainer_or_config : BaseTrainer | ModelConfig) -> None:
        self.bound_with(trainer_or_config)
        self.folder_preds.mkdir(exist_ok=True , parents=True)
        self.folder_avg_preds.mkdir(exist_ok=True , parents=True)

    def __repr__(self):
        return f'{self.__class__.__name__}(trainer={self.trainer})'

    @property
    def pred_idx(self):
        """unique index for each prediction batch"""
        return f'{self.model_num}.{self.model_date}.{self.model_submodel}.{self.batch_idx}'

    @cached_property
    def pred_dict(self) -> dict[str,pd.DataFrame]:
        """stored predictions for each batch"""
        return {}

    @cached_property
    def resumed_models(self) -> pd.DataFrame:
        """
        Resumed models in testing, these models are finished and will not be loaded and tested again. 
        Columns: model_num, model_date

        The dataframe is the same for both scenarios of resuming testing from last pred date and last model date.
        """
        return pd.DataFrame(columns = ['model_num' , 'model_date'])
    @property
    def snap_folder(self) -> Path: 
        """folder to save model predictions"""
        return self.base_path.snapshot('pred_recorder')
    @property
    def folder_preds(self) -> Path:
        """folder to save model predictions"""
        return self.snap_folder.joinpath('preds')
    @property
    def folder_avg_preds(self) -> Path:
        """folder to save averaged model predictions"""
        return self.snap_folder.joinpath('avg_preds')
    @property
    def folder_records(self) -> Path:
        """folder to save prediction records, such as missing pred dates"""
        return self.snap_folder.joinpath('records')
    @property
    def min_test_date(self) -> int:
        """minimum test date , considering the data and config"""
        return max(self.data.min_test_date , self.config.beg_date)
    @property
    def max_test_date(self) -> int:
        """maximum test date , considering the data and config"""
        return max(self.data.max_test_date , self.config.end_date)

    def save_preds(self , df : pd.DataFrame , model_date : int , model_num : int , append = False , async_save : bool = False):
        if df.empty:
            return
        
        old_path = [path for path in self.folder_preds.glob('*.feather') if path.name.startswith(f'{model_num}.{model_date}.')]
        assert len(old_path) <= 1 , f'Multiple old paths found for model {model_num} at date {model_date}: {old_path}'
        if old_path and append:
            old_df = DB.load_df(old_path[0])
            df = pd.concat([old_df , df]).drop_duplicates(subset=self.PRED_KEYS + self.PRED_IDXS , keep='last').sort_values(by=self.PRED_KEYS + self.PRED_IDXS)
            
        min_pred_date , max_pred_date = df['date'].min() , df['date'].max()
        path = self.folder_preds.joinpath(f'{model_num}.{model_date}.{min_pred_date}.{max_pred_date}.feather')
        self.update_missing_pred_dates(CALENDAR.range(min_pred_date , max_pred_date , 'td') , df['date'].unique())
        if old_path and path != old_path[0]:
            Path(old_path[0]).unlink()
        if async_save:
            AsyncSaver.df(df , path , future_group = 'pred_recorder' , vb_level = 'max')
        else:
            DB.save_df(df , path , overwrite = True , vb_level = 'max')

    def save_avg_preds(self , model_date : int , async_save : bool = False):
        AsyncSaver.wait_all('pred_recorder')
        pred_paths = [path for path in self.folder_preds.glob('*.feather') if path.name.split('.')[1] == str(model_date)]
        df = DB.load_df(pred_paths , key_column = None)
        if df.empty:
            return
        df = df.groupby(['model_date' , 'submodel' , 'secid' , 'date'])[['pred' , 'label']].mean().reset_index()
        min_pred_date , max_pred_date = df['date'].min() , df['date'].max()
        path = self.folder_avg_preds.joinpath(f'{model_date}.{min_pred_date}.{max_pred_date}.feather')
        if async_save:
            AsyncSaver.df(df , path , future_group = 'pred_recorder' , vb_level = 'max')
        else:
            DB.save_df(df , path , overwrite = True , vb_level = 'max')

    def update_avg_preds(self , pred_df : pd.DataFrame):
        if pred_df.empty:
            return pred_df.drop(columns = ['model_num'] , errors='ignore')
        avg_df = pred_df.groupby(['model_date' , 'submodel' , 'secid' , 'date'])[['pred' , 'label']].mean().reset_index()
        for model_date in avg_df['model_date'].unique():
            avg_paths = [path for path in self.folder_avg_preds.glob('*.feather') if path.name.startswith(f'{model_date}.')]
            assert len(avg_paths) <= 1 , f'Multiple old paths found for model {model_date}: {avg_paths}'
            if avg_paths:
                old_path = avg_paths[0]
                old_df = DB.load_df(old_path)
            else:
                old_path = None
                old_df = pd.DataFrame()
            new_df = avg_df.query('model_date == @model_date')
            if not old_df.empty:
                new_df = pd.concat([old_df , new_df]).drop_duplicates(subset=['model_date' , 'submodel' , 'secid' , 'date'] , keep='last').\
                    sort_values(by=['model_date' , 'submodel' , 'secid' , 'date']).reset_index(drop=True)

            min_pred_date , max_pred_date = new_df['date'].min() , new_df['date'].max()
            new_path = self.folder_avg_preds.joinpath(f'{model_date}.{min_pred_date}.{max_pred_date}.feather')
            if old_path and old_path != new_path:
                old_path.unlink()
            DB.save_df(new_df , new_path , overwrite = True , vb_level = 'max')
            
        self.logger.note(f'Updated avg preds for pred dates {pred_df["date"].unique()}')
        return avg_df

    def archive_model_records(self):
        records : list[tuple[int,int]] = []
        for model_num , model_date , _ , _ in self.config.base_path.iter_model_archives():
            records.append((model_num , model_date))
        df = pd.DataFrame(records , columns = ['model_num' , 'model_date']) if records else pd.DataFrame(columns = ['model_num' , 'model_date']).astype(int)
        df = df.drop_duplicates().sort_values(by=['model_num' , 'model_date'])
        df['next_model_date'] = df.groupby('model_num')['model_date'].shift(-1)
        return df

    def pred_records(self): 
        """model_date/model_num of saved predictions"""
        return pd.DataFrame([[path , *path.name.split('.')[:4]] for path in self.folder_preds.glob('*.feather')], columns = ['path' , 'model_num' , 'model_date' , 'min_pred_date' , 'max_pred_date']).\
            astype({'model_num' : int , 'model_date' : int , 'min_pred_date' : int , 'max_pred_date' : int})

    def avg_pred_records(self):
        return pd.DataFrame([[path , *path.name.split('.')[:3]] for path in self.folder_avg_preds.glob('*.feather')], columns = ['path' , 'model_date' , 'min_pred_date' , 'max_pred_date']).\
            astype({'model_date' : int , 'min_pred_date' : int , 'max_pred_date' : int})

    def update_missing_pred_dates(self , required_dates : np.ndarray | list[int] , existing_dates : np.ndarray | list[int] | None = None):
        """log missing prediction dates in records folder"""
        existing_missing_dates = self.get_missing_pred_dates()
        missing_dates = np.union1d(required_dates , existing_missing_dates)
        if existing_dates is not None:
            missing_dates = np.setdiff1d(missing_dates , existing_dates)

        PATH.dump_json({'missing_dates' : missing_dates.tolist()} , 
                       self.folder_records.joinpath('missing_pred_dates.json') , overwrite = True)

    def get_missing_pred_dates(self) -> np.ndarray:
        """get missing prediction dates from records folder"""
        try:
            path = self.folder_records.joinpath('missing_pred_dates.json')
            if not path.exists():
                return np.array([] , dtype=int)
            return np.array(PATH.read_json(path)['missing_dates'])
        except Exception:
            return np.array([] , dtype=int)

    @cached_property
    def retrained_models(self) -> list[tuple[int,int]]:
        """retrained models for resumed testing , must be tested"""
        return []

    @classmethod
    def empty_preds(cls) -> pd.DataFrame:
        """empty predictions dataframe"""
        return pd.DataFrame(columns = cls.PRED_KEYS + cls.PRED_IDXS + cls.PRED_COLS)

    def append_retrained_model(self):
        """
        append retrained model to retrained_models list
        """
        self.retrained_models.append((self.model_date , self.model_num))

    @cached_property
    def has_purged_preds(self) -> bool:
        return False

    def purge_retrained_model_preds(self):
        """purge past predictions when trained new models"""
        if not self.retrained_models:
            self.logger.stdout(f'No retrained models, no purge needed')
            return
        min_retrained_model_date = min([model_date for model_date , _ in self.retrained_models])
        pred_records = self.pred_records()
        purge_models = pred_records.query('model_date >= @min_retrained_model_date')
        trim_models = pred_records.query('model_date < @min_retrained_model_date and max_pred_date > @min_retrained_model_date')
        if not purge_models.empty or not trim_models.empty:
            purge_info = f'Purged saved predictions after retrained model date {min_retrained_model_date}'
            if not purge_models.empty:
                purge_info += f', {len(purge_models)} models(date/num) purged'
            if not trim_models.empty:
                purge_info += f', {len(trim_models)} models(date/num) trimed'
            
            for _ , (model_date , model_num , path) in trim_models.loc[:,['model_date' , 'model_num' , 'path']].iterrows():
                df = DB.load_df(path).query('date <= @min_retrained_model_date')
                Path(path).unlink()
                self.save_preds(df , model_date , model_num)
                
            self.has_purged_preds = True
        else:
            purge_info = f'{len(self.retrained_models)} models retrained, but no pred need to be purged'
        self.logger.stdout(purge_info)

    def purge_outdated_model_preds(self):
        archive_records = self.archive_model_records()
        pred_records = self.pred_records()
        new_pred_records = archive_records.merge(pred_records , on=['model_num' , 'model_date'] , how='outer')
        new_pred_records['next_model_date'] = new_pred_records['next_model_date'].fillna(99991231)
        df = new_pred_records.query('min_pred_date <= model_date or max_pred_date > next_model_date')
        if df.empty:
            self.logger.stdout(f'No outdated predictions found, no purge needed')
            return

        purge_info = f'Purged outdated predictions, {len(df)} models(date/num) partially purged :'
        self.logger.display(df , caption = purge_info , vb_level = self.vb_level)
        for _ , (model_date , model_num , path , next_model_date) in df.loc[:,['model_date' , 'model_num' , 'path' , 'next_model_date']].iterrows():
            df = DB.load_df(path).query('date <= @next_model_date and date >= @model_date')
            Path(path).unlink()
            self.save_preds(df , model_date , model_num)
        self.has_purged_preds = True

    def purge_duplicated_model_preds(self):
        """purge duplicated model predictions"""
        AsyncSaver.wait_all('pred_recorder')
        pred_records = self.pred_records()
        pred_records = pred_records.sort_values(by = ['model_date' , 'model_num' , 'max_pred_date' , 'min_pred_date'] , ascending = [True , True , False , True])
        obsolete_records = pred_records[pred_records.duplicated(subset = ['model_date' , 'model_num'])]

        avg_pred_records = self.avg_pred_records()
        avg_pred_records = avg_pred_records.sort_values(by = ['model_date' , 'max_pred_date' , 'min_pred_date'] , ascending = [True , False , True])
        obsolete_avg_records = avg_pred_records[avg_pred_records.duplicated(subset = ['model_date'])]
        if obsolete_records.empty and obsolete_avg_records.empty:
            return
        purge_info = f'Purged obsolete predictions'
        if not obsolete_records.empty:
            purge_info += f', {len(obsolete_records)} model preds'
            for path in obsolete_records['path']:
                Path(path).unlink()
        if not obsolete_avg_records.empty:
            purge_info += f', {len(obsolete_avg_records)} avg model preds'
            for path in obsolete_avg_records['path']:
                Path(path).unlink()
        purge_info += f' deleted!'
        self.logger.stdout(purge_info)
        self.has_purged_preds = True

    def setup_resuming_status(self):
        """
        setup resuming status for previous saved predictions
        notes:
        - only resume predictions before the last model date if resume option is 'last_model_date'
        - only resume predictions with all submodels
        """ 
        self.resume_info = ''
        if not self.config.is_resuming or not Const.Model.resume_test:
            return
        
        pred_records = self.pred_records().query('max_pred_date >= @self.min_test_date & min_pred_date <= @self.max_test_date')
        if pred_records.empty:
            self.resume_info = f', no saved preds found'
            self.config.resumed_max_pred_date = 19000101
            return

        closest_model_date = pred_records.groupby('model_num')['model_date'].max().min() # noqa: F841
        min_pred_date = pred_records.groupby('model_num')['min_pred_date'].min().max()
        if self.min_test_date < CALENDAR.td(min_pred_date , -5): # leave 5 days buffer for resume testing model
            self.resume_info = f', but new test start {self.min_test_date} is too many days earlier than saved preds {min_pred_date}, forfeiting resume preds'
            self.config.resumed_max_pred_date = 19000101
            return
 
        missing_dates = CALENDAR.slice(self.get_missing_pred_dates() , self.min_test_date , self.max_test_date)
        max_pred_date = min([
            pred_records.groupby('model_num')['max_pred_date'].max().min(),
            CALENDAR.td(missing_dates.min() , -1).td if len(missing_dates) > 0 else self.max_test_date,
            self.max_test_date
        ])
        
        valid_models = pred_records.query('model_date <= @closest_model_date')
        prev_models = valid_models.query('model_date < @closest_model_date')
        resumed_models = prev_models.query('max_pred_date <= @max_pred_date')[['model_date' , 'model_num']].reset_index(drop=True)
        self.resumed_models = resumed_models
        if Const.Model.resume_test_start == 'last_model_date':
            if not prev_models.empty:
                max_pred_date = min(max_pred_date , prev_models['max_pred_date'].max())
            self.config.resumed_max_pred_date = max_pred_date
            self.resume_info = f', recognize past saved preds before model date {self.resumed_models["model_date"].max()} and prediction date {self.config.resumed_max_pred_date}'
        elif Const.Model.resume_test_start == 'last_pred_date':
            self.config.resumed_max_pred_date = max_pred_date
            self.resume_info = f', recognize past saved preds before prediction date {self.config.resumed_max_pred_date}'
    
    def append_batch_preds(self):
        if self.pred_idx in self.pred_dict.keys() or self.batch_output.empty: 
            return
        df = self.batch_data.pred_df().dropna(how = 'all').query('date in @self.data.test_full_dates')
        if df.empty:
            return
        if (which_output := self.trainer.model_param.get('which_output' , 0)) is None:
            df['pred'] = df.loc[:,[col for col in df.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            df['pred'] = df[f'pred.{which_output}']
        df = df.assign(model_num = self.model_num , submodel = self.model_submodel , model_date = self.model_date , batch_idx = self.batch_idx)
        df = df.loc[:,self.PRED_KEYS + self.PRED_IDXS + self.PRED_COLS]
        self.pred_dict[self.pred_idx] = df

    def collect_model_preds(self):
        if not self.pred_dict:
            return self.empty_preds()
        new_preds = pd.concat(self.pred_dict.values())
        self.save_preds(new_preds , self.model_date , self.model_num , append = True , async_save = True)
        self.pred_dict.clear()

    def collect_avg_preds(self):
        self.save_avg_preds(self.model_date , async_save = True)
        
    def get_preds(self , pred_dates : np.ndarray , model_num : int | None = None , closest : bool = False) -> pd.DataFrame:
        # maybe give start and end dates to the function? so that analysis can start from last analysis date, instead of last pred date
        if len(pred_dates) == 0:
            return self.empty_preds()
        pred_records = self.pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        if model_num is not None:
            pred_records = pred_records.query('model_num == @model_num')
        if pred_records.empty:
            self.logger.error(f'No pred records found for test dates {pred_dates}')
            if closest:
                pred_records = self.pred_records().query('max_pred_date <= @pred_dates.min()')
                closest_pred_date = pred_records['max_pred_date'].max() # noqa: F841
                pred_records = pred_records.query('max_pred_date == @closest_pred_date')
                df = DB.load_df({path:path for path in pred_records['path']} , key_column = 'path')
            else:
                df = self.empty_preds()
        else:
            df = DB.load_df({path:path for path in pred_records['path']} , key_column = 'path')
            self.update_missing_pred_dates(pred_dates , df['date'].unique())
        df = self.try_fill_na_label(df , try_rewrite = True).drop(columns = ['path'] , errors = 'ignore')
        df = df.query('date in @pred_dates').reset_index(drop = True)
        return df

    def get_avg_preds(self , pred_dates : np.ndarray , closest : bool = False) -> pd.DataFrame:
        # maybe give start and end dates to the function? so that analysis can start from last analysis date, instead of last pred date
        if len(pred_dates) == 0:
            return self.empty_preds()
        avg_pred_records = self.avg_pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        pred_records = self.pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        [self.save_avg_preds(model_date) for model_date in np.setdiff1d(pred_records['model_date'] , avg_pred_records['model_date'])]
        
        avg_pred_records = self.avg_pred_records().query('min_pred_date <= @pred_dates.max() & max_pred_date >= @pred_dates.min()')
        if avg_pred_records.empty:
            self.logger.error(f'No avg pred records found for test dates {pred_dates}')
            pred_df = self.get_preds(pred_dates , closest = closest)
            df = self.update_avg_preds(pred_df)
            if df.empty:
                if closest:
                    avg_pred_records = self.avg_pred_records().query('max_pred_date <= @pred_dates.min()')
                    closest_avg_pred_date = avg_pred_records['max_pred_date'].max() # noqa: F841
                    avg_pred_records = avg_pred_records.query('max_pred_date == @closest_avg_pred_date')
                    df = DB.load_df({path:path for path in avg_pred_records['path']} , key_column = 'path')
                else:
                    df = self.empty_preds()
        else:
            df = DB.load_df({path:path for path in avg_pred_records['path']} , key_column = 'path')
        df = self.try_fill_na_label(df , try_rewrite = True).drop(columns = ['path'] , errors = 'ignore')
        df = df.query('date in @pred_dates').reset_index(drop = True)
        return df

    def try_fill_na_label(self , pred_df : pd.DataFrame , try_rewrite : bool = True) -> pd.DataFrame:
        if pred_df.empty:
            return pred_df
        new_df = pred_df.assign(na_label = pred_df['label'].isna()).groupby('date')['na_label'].all().reset_index()
        na_label_dates = new_df.query('na_label == True')['date'].unique()
        if len(na_label_dates) == 0:
            return pred_df
        new_df = self.data.y_label(na_label_dates).rename(columns = {'label' : 'new_label'})
        pred_df = pred_df.merge(new_df , on = ['secid' , 'date'] , how = 'left')
        pred_df['label'] = pred_df['label'].where(pred_df['label'].notna(), pred_df['new_label'])
        if try_rewrite and 'path' in pred_df.columns:
            for path in pred_df['path'].unique():
                subdf = pred_df.query('path == @path').drop(columns = ['path']).reset_index(drop = True)
                if not subdf.empty and subdf['new_label'].notna().any():
                    self.logger.stdout(f'rewriting na label for {path}')
                    DB.save_df(subdf.drop(columns = ['new_label']) , path , overwrite = True)
        return pred_df.drop(columns = ['new_label']).reset_index(drop = True)

    def on_configure_model(self):
        self.setup_resuming_status()
        if self.resume_info:
            self.logger.stdout(f'Resume testing {self.resume_info}')

    def on_fit_model_end(self):
        self.append_retrained_model()

    def on_fit_end(self):
        self.purge_retrained_model_preds()

    def on_test_start(self):
        self.purge_outdated_model_preds()
        if self.has_purged_preds:
            self.setup_resuming_status()
            self.has_purged_preds = False
        if self.resume_info:
            self.logger.stdout(f'Resume testing {self.resume_info}' , ind = 1)
    
    def on_test_batch_end(self): 
        self.append_batch_preds()

    def on_test_model_end(self):
        self.collect_model_preds()

    def on_test_model_date_end(self):
        self.collect_avg_preds()

    def on_test_end(self):
        self.purge_duplicated_model_preds()
        avg_pred_records = self.avg_pred_records()
        if not avg_pred_records.empty:
            self.logger.stdout(f'avg model preds updated to {avg_pred_records["max_pred_date"].max()}')