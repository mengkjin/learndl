import joblib
import pandas as pd
import torch

from datetime import datetime
from pathlib import Path
from typing import Sequence , Literal

from src.proj import PATH
from src.proj import Logger
from src.proj.func import torch_load
from src.res.deap.param import gpDefaults
from .syntax import SyntaxRecord
from .status import gpStatus

class gpLogger:
    """process logger for genetic programming"""
    _instance = None
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , job_dir : Path | str | None = None , status : gpStatus | None = None , *args , **kwargs) -> None:
        self.initiate(job_dir , status , *args , **kwargs)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'job_dir')

    def initiate(self , job_dir : Path | str | None = None , status : gpStatus | None = None) -> None:
        if self.initiated:
            return

        self.job_dir = gpDefaults.dir_pop.joinpath('bendi') if job_dir is None else Path(job_dir)
        self.status = status if status is not None else gpStatus(0 , 0)
        
        self.dir_log = self.job_dir.joinpath('logbook')
        self.dir_sku = self.job_dir.joinpath('skuname')
        self.dir_res = self.job_dir.joinpath('labels_res')
        self.dir_elt = self.job_dir.joinpath('elites')
        self.dir_neu = self.job_dir.joinpath('neutra')
        for dir in [self.dir_log , self.dir_sku , self.dir_res , self.dir_elt , self.dir_neu]:
            dir.mkdir(parents=True, exist_ok=True)

        self.paths : dict[str, Path] = {
            'elitelog' : self.job_dir.joinpath('elite_log.csv') ,
            'hoflog' : self.job_dir.joinpath('hof_log.csv') ,
            'runtime' : self.job_dir.joinpath('runtime.csv') ,
            'params' : self.job_dir.joinpath('params.yaml') ,
            'df_axis' : self.job_dir.joinpath('df_axis.pt') ,
            'historybook' : self.job_dir.joinpath('historybook.pkl')
        }
        self.df_axis = {}

    def get_record_basename(self , i_iter = 0 , i_gen = -1):
        assert i_iter >= 0 , f'i_iter must be greater than 0, but got {i_iter}'
        assert i_gen >= -1 , f'i_gen must be greater than -1, but got {i_gen}'
        if i_gen == -1 and i_iter == 0:
            return ''
        elif i_gen == -1:
            return f'iter{i_iter - 1}_overall'
        else:
            return f'iter{i_iter}_gen{i_gen}'

    @property
    def train(self) -> bool:
        return self.status.train

    @property
    def final_record_basename(self):
        return self.get_record_basename(self.status.i_iter + 1 , -1)

    @property
    def current_record_basename(self):
        return self.get_record_basename(self.status.i_iter , self.status.i_gen)

    @property
    def previous_record_basename(self):
        return self.get_record_basename(self.status.i_iter , self.status.i_gen - 1)

    def load_generation(self , i_gen : int | None = None , **kwargs) -> tuple[list[SyntaxRecord] , list[SyntaxRecord] , list[str]]:
        basename = self.previous_record_basename if i_gen is None else self.get_record_basename(self.status.i_iter , i_gen - 1)
        if not basename:
            return [] , [] , []

        if self.dir_log.joinpath(f'{basename}.pkl').exists():
            log = joblib.load(self.dir_log.joinpath(f'{basename}.pkl'))
            pop = [ind for ind in log['population']] 
            hof = [ind for ind in log['halloffame']]
            fbd = [ind for ind in log['forbidden']] 
            return pop , hof , fbd
        else:
            raise Exception(f'{self.dir_log.joinpath(f"{basename}.pkl")} does not exists!')

    def dump_generation(self , population : Sequence , halloffame : Sequence , forbidden : Sequence , overall = False , **kwargs):
        # save type: list of SyntaxRecord for population and halloffame, list of str for forbidden
        dumps = {
            'i_iter' : self.status.i_iter,
            'i_gen' : self.status.i_gen,
            'population' : population,
            'halloffame' : halloffame,
            'forbidden' : forbidden,
            **kwargs
        }
        if overall:
            assert not population and not halloffame , 'overall dump should not have population & halloffame, only forbidden'
            assert self.status.i_gen >= self.status.n_gen - 1 , f'i_iter {self.status.i_iter} has i_gen {self.status.i_gen} not reach end state {self.status.n_gen - 1}'
            path = self.dir_log.joinpath(f'{self.final_record_basename}.pkl')
        else:
            path = self.dir_log.joinpath(f'{self.current_record_basename}.pkl')
        joblib.dump(dumps , path)
        Logger.success(f'Generation {self.status.i_gen} Logbook Saved to "{path}"' , indent = 1)
        return self

    def load_historybook(self , **kwargs) -> dict[str, SyntaxRecord]:
        if not self.paths['historybook'].exists():
            self.historybook = {}
        else:
            self.historybook = joblib.load(self.paths['historybook'])
        return self.historybook

    def update_historybook(self , population : Sequence[SyntaxRecord] , **kwargs):
        self.historybook.update({str(ind) : ind for ind in population})
        return self

    def dump_historybook(self , **kwargs):
        joblib.dump(self.historybook , self.paths['historybook'])
        Logger.success(f'Historybook Saved to "{self.paths['historybook']}"')
        return self

    def update_sku(self , syntax : str , pool_skuname : str):
        poolid = int(pool_skuname.split('_')[-1])
        if poolid % 100 == 0:
            start_time_sku = datetime.now()
            output_path = self.dir_sku.joinpath(f'z_{pool_skuname}.txt')
            with open(output_path, 'w', encoding='utf-8') as file1:
                file1.write(f'{syntax}\n start_time {start_time_sku}')

    def state_data_path(self , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'df_axis' , 'runtime' , 'params'] , **kwargs) -> Path:
        if key in ['res' , 'neu' , 'elt']:
            return getattr(self , f'dir_{key}').joinpath(f'iter{kwargs['i_iter']}.pt')
        elif key in ['elitelog' , 'hoflog' , 'runtime']:
            return self.job_dir.joinpath(f'{key}.csv')
        elif key == 'df_axis':
            return self.job_dir.joinpath('df_axis.pt')
        elif key == 'params':
            return self.job_dir.joinpath('params.yaml')
        else:
            raise Exception(key)
    
    def load_state(self , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'df_axis' , 'runtime'] , i_iter , device = None):
        path = self.state_data_path(key , i_iter = i_iter)
        if key in ['res' , 'neu' , 'elt']:
            return torch_load(path).to(device)
        elif key in ['elitelog' , 'hoflog']:
            i_iter = i_iter if i_iter >= 0 else 99
            return pd.read_csv(path,index_col=0).query('i_iter < @i_iter') if path.exists() else pd.DataFrame()
        elif key == 'runtime':
            return pd.read_csv(path,index_col=0) if path.exists() else pd.DataFrame()
        elif key == 'df_axis':
            self.df_axis = torch_load(path)
            return self.df_axis
        else:
            raise Exception(key)

    def save_state(self , data , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'df_axis' , 'runtime' , 'params'] , i_iter , **kwargs):
        if not self.train:
            return

        path = self.state_data_path(key , i_iter = i_iter)
        if key in ['res' , 'neu' , 'elt']:
            torch.save(data , path)
        elif key in ['elitelog' , 'hoflog' , 'runtime']:
            assert isinstance(data , pd.DataFrame) , data
            data.reset_index(drop=True).to_csv(path)
        elif key == 'df_axis':
            assert isinstance(data , dict) , data
            torch.save(data , path)
            self.df_axis = data
        elif key == 'params':
            assert isinstance(data , dict) , data
            PATH.dump_yaml(data , path)
        else:
            raise Exception(key , data)
            
    def load_states(self, keys , **kwargs):
        return [self.load_state(key , **kwargs) for key in keys]
    
    def save_states(self, datas , **kwargs):
        return [self.save_state(data , key , **kwargs) for key , data in datas.items()]