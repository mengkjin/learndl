import joblib
import pandas as pd
import numpy as np
import torch

from datetime import datetime
from pathlib import Path
from typing import Sequence , Literal , Any

from src.proj import PATH , Proj , Logger
from src.proj.core import strPath
from src.proj.util import torch_load
from src.res.gp.param import gpDefaults
from .syntax import SyntaxRecord
from .status import gpStatus

class gpLogger:
    """process logger for genetic programming"""
    _instance = None
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , job_dir : strPath | None = None , status : gpStatus | None = None , vb_level : Any = 2 , *args , **kwargs) -> None:
        self.initiate(job_dir , status , vb_level , *args , **kwargs)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'job_dir')

    def initiate(self , job_dir : strPath | None = None , status : gpStatus | None = None , vb_level : Any = 2) -> None:
        if self.initiated:
            return

        self.job_dir = gpDefaults.dir_result.joinpath('bendi') if job_dir is None else Path(job_dir)
        self.status = status if status is not None else gpStatus(0 , 0)
        self.vb_level = Proj.vb(vb_level)

        for dir in [self.dir_logbook , self.dir_tensors , self.dir_records]:
            dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir_logbook(self):
        return self.job_dir.joinpath('logbook')
    @property
    def dir_tensors(self):
        return self.job_dir.joinpath('tensors')
    @property
    def dir_records(self):
        return self.job_dir.joinpath('records')

    @property
    def path_historybook(self):
        return self.dir_logbook.joinpath('historybook.pkl')

    def get_logbook_basename(self , i_iter = 0 , i_gen = -1):
        assert i_iter >= 0 , f'i_iter must be greater than 0, but got {i_iter}'
        assert i_gen >= -1 , f'i_gen must be greater than -1, but got {i_gen}'
        if i_gen == -1 and i_iter == 0:
            return ''
        elif i_gen == -1:
            return f'iter{i_iter - 1}_overall'
        else:
            return f'iter{i_iter}_gen{i_gen}'

    @property
    def final_logbook_name(self):
        return self.get_logbook_basename(self.status.i_iter + 1 , -1)

    @property
    def current_logbook_name(self):
        return self.get_logbook_basename(self.status.i_iter , self.status.i_gen)

    @property
    def previous_logbook_name(self):
        return self.get_logbook_basename(self.status.i_iter , self.status.i_gen - 1)

    def load_generation(self , i_gen : int | None = None , **kwargs) -> tuple[list[SyntaxRecord] , list[SyntaxRecord] , list[str]]:
        basename = self.previous_logbook_name if i_gen is None else self.get_logbook_basename(self.status.i_iter , i_gen - 1)
        if not basename:
            return [] , [] , []

        if self.dir_logbook.joinpath(f'{basename}.pkl').exists():
            log = joblib.load(self.dir_logbook.joinpath(f'{basename}.pkl'))
            pop = [ind for ind in log['population']] 
            hof = [ind for ind in log['halloffame']]
            fbd = [ind for ind in log['forbidden']] 
            return pop , hof , fbd
        else:
            raise Exception(f'{self.dir_logbook.joinpath(f"{basename}.pkl")} does not exists!')

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
            path = self.dir_logbook.joinpath(f'{self.final_logbook_name}.pkl')
        else:
            path = self.dir_logbook.joinpath(f'{self.current_logbook_name}.pkl')
        joblib.dump(dumps , path)
        Logger.success(f'Generation {self.status.i_gen} Logbook Saved to "{path}"' , indent = 1 , vb_level = self.vb_level)
        return self

    def load_historybook(self , **kwargs) -> dict[str, SyntaxRecord]:
        historybook = None
        if self.path_historybook.exists():
            historybook = joblib.load(self.path_historybook)
        return {} if historybook is None else historybook

    def dump_historybook(self , historybook : dict[str, SyntaxRecord] | None = None , **kwargs):
        joblib.dump(historybook , self.path_historybook)
        Logger.success(f'Historybook Saved to {self.path_historybook}' , vb_level = self.vb_level)
        return self

    def update_sku(self , syntax : str , pool_skuname : str):
        poolid = int(pool_skuname.split('_')[-1])
        if poolid % 100 == 0:
            start_time_sku = datetime.now()
            output_path = self.dir_records.joinpath(f'skuname_{pool_skuname}.txt')
            with open(output_path, 'w', encoding='utf-8') as file1:
                file1.write(f'{syntax}\n start_time {start_time_sku}')

    def state_data_path(self , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'secid' , 'date' , 'runtime' , 'params'] , 
                        i_iter : int | None = None , **kwargs) -> Path:
        match key:
            case 'res' | 'neu' | 'elt':
                assert i_iter is not None and i_iter >= 0 , f'i_iter must be greater than 0, but got {i_iter}'
                prefix = {
                    'res' : 'labels_res',
                    'neu' : 'neutra',
                    'elt' : 'elites',
                }[key]
                return self.dir_tensors.joinpath(f'{prefix}_iter{i_iter}.pt')
            case 'elitelog' | 'hoflog' | 'runtime':
                return self.dir_records.joinpath(f'{key}.csv')
            case 'secid' | 'date':
                return self.dir_records.joinpath(f'{key}.pt')
            case 'params':
                return self.dir_records.joinpath('params.yaml')
            case _:
                raise Exception(key)
    
    def load_state(self , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'secid' , 'date' , 'runtime' , 'params'] , 
                   i_iter : int | None = None , device : torch.device | None = None) -> Any:
        path = self.state_data_path(key , i_iter = i_iter)
        match key:
            case 'res' | 'neu' | 'elt':
                data = torch_load(path)
                if isinstance(data , torch.Tensor):
                    data = data.to(device)
                return data
            case 'elitelog' | 'hoflog' | 'runtime':
                df = pd.read_csv(path,index_col=0) if path.exists() else pd.DataFrame()
                if not df.empty and i_iter is not None and 'i_iter' in df.columns:
                    df = df.query('i_iter <= @i_iter')
                return df
            case 'secid' | 'date':
                return torch_load(path)
            case 'params':
                return PATH.read_yaml(path)
            case _:
                raise Exception(key)

    def save_state(self , key : Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'secid' , 'date' , 'runtime' , 'params'] , 
                   data : Any , i_iter : int | None = None, **kwargs):
        if not self.status.train:
            return
        path = self.state_data_path(key , i_iter = i_iter)
        match key:
            case 'res' | 'neu' | 'elt':
                assert data is None or isinstance(data , torch.Tensor) , data
                torch.save(data , path)
            case 'elitelog' | 'hoflog' | 'runtime':
                assert isinstance(data , pd.DataFrame) , data
                data.reset_index(drop=True).to_csv(path)
            case 'secid' | 'date':
                assert isinstance(data , np.ndarray) , data
                torch.save(data , path)
            case 'params':
                assert isinstance(data , dict) , data
                if path.exists():
                    Logger.alert1(f'{path} already exists, it will be overwritten' , vb_level = self.vb_level)
                    path.unlink()
                PATH.dump_yaml(data , path)
            case _:
                raise Exception(key)
            
    def load_states(self, keys , **kwargs):
        return [self.load_state(key , **kwargs) for key in keys]
    
    def save_states(self, datas : dict[Literal['res' , 'neu' , 'elt' , 'elitelog' , 'hoflog' , 'secid' , 'date' , 'runtime' , 'params'] , Any], **kwargs):
        return [self.save_state(key , data , **kwargs) for key , data in datas.items()]