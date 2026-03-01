import joblib
import pandas as pd
import torch

from datetime import datetime
from deap import tools
from pathlib import Path
from typing import Sequence , Literal

from src.proj import PATH
from src.proj.func import torch_load
from src.res.deap.param import gpDefaults
from .syntax import BaseIndividual , SyntaxRecord
from .status import gpStatus

class gpLogger:
    _instance = None
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , job_dir : Path | str | None = None , status : gpStatus | None = None , *args , **kwargs) -> None:
        self.initialize(job_dir , status , *args , **kwargs)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'job_dir')

    def initialize(self , job_dir : Path | str | None = None , status : gpStatus | None = None) -> None:
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
            'df_axis' : self.job_dir.joinpath('df_axis.pt')
        }
        
        self.df_axis = {}

    def get_record_basename(self , i_iter = 0 , i_gen = -1):
        assert i_iter >= 0 , f'i_iter must be greater than 0, but got {i_iter}'
        iter_str = f'iter{i_iter}'
        gen_str  = 'overall' if i_gen < 0 else f'gen{i_gen}'
        return f'{iter_str}_{gen_str}'

    @property
    def final_record_basename(self):
        return self.get_record_basename(self.status.n_iter , -1)

    @property
    def current_record_basename(self):
        return self.get_record_basename(self.status.i_iter , self.status.i_gen)

    @property
    def previous_record_basename(self):
        if self.status.i_iter == 0 and self.status.i_gen == 0:
            return ''
        elif self.status.i_gen == 0:
            return self.get_record_basename(self.status.i_iter - 1 , -1)
        else:
            return self.get_record_basename(self.status.i_iter , self.status.i_gen - 1)

    def load_generation(self , **kwargs) -> tuple[list[SyntaxRecord] , list[SyntaxRecord] , list[str]]:
        self.logbook = tools.Logbook()   
        basename = self.previous_record_basename
        if not basename:
            return [] , [] , []

        if self.dir_log.joinpath(f'{basename}.pkl').exists():
            log = joblib.load(self.dir_log.joinpath(f'{basename}.pkl'))
            self.logbook.record(**log)
            if self.status.i_gen >= 0:
                pop = [ind for ind in log['population']] 
                hof = [ind for ind in log['halloffame']]
            else:
                pop , hof = [] , []
            fbd = [ind for ind in log['forbidden']] 
            return pop , hof , fbd
        else:
            raise Exception(f'{self.dir_log.joinpath(f"{basename}.pkl")} does not exists!')

    def dump_generation(self , population : Sequence , halloffame : Sequence , forbidden : Sequence , **kwargs):
        # save type: list of SyntaxRecord for population and halloffame, list of str for forbidden
        self.logbook.record(i_gen = self.status.i_gen , population = population, 
                            halloffame = halloffame, forbidden = forbidden, 
                            **kwargs)

        joblib.dump(self.logbook[-1] , self.dir_log.joinpath(f'{self.current_record_basename}.pkl'))
        if self.status.i_gen == self.status.n_gen:
            joblib.dump(self.logbook[-1] , self.dir_log.joinpath(f'{self.final_record_basename}.pkl'))
        return self

    def update_sku(self , individual , pool_skuname : str):
        poolid = int(pool_skuname.split('_')[-1])
        if poolid % 100 == 0:
            start_time_sku = datetime.now()
            output_path = self.dir_sku.joinpath(f'z_{pool_skuname}.txt')
            with open(output_path, 'w', encoding='utf-8') as file1:
                file1.write(f'{BaseIndividual.trim_syntax(individual)}\n start_time {start_time_sku}')

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
            return pd.read_csv(path,index_col=0).query('i_iter < @i_iter') if path.exists() else pd.DataFrame()
        elif key == 'runtime':
            return pd.read_csv(path,index_col=0) if path.exists() else pd.DataFrame()
        elif key == 'df_axis':
            self.df_axis = torch_load(path)
            return self.df_axis
        else:
            raise Exception(key)

    def save_state(self , data , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'df_axis' , 'runtime' , 'params'] , i_iter , **kwargs):
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