import joblib , re
import pandas as pd
import torch

from datetime import datetime
from deap import tools
from pathlib import Path
from typing import Any , Sequence

from src.proj import Logger
from src.proj.func import torch_load
from src.res.deap.env import gpDefaults
from .syntax import SyntaxControl
from .toolbox import BaseToolbox

class gpLogger:
    def __init__(self , job_dir : Path | str | None = None) -> None:
        self.job_dir = gpDefaults.dir_pop.joinpath('bendi') if job_dir is None else Path(job_dir)
        
        self.dir_log = self.job_dir.joinpath('logbook')
        self.dir_sku = self.job_dir.joinpath('skuname')
        self.dir_pqt = self.job_dir.joinpath('parquet')
        self.dir_res = self.job_dir.joinpath('labels_res')
        self.dir_elt = self.job_dir.joinpath('elites')
        self.dir_neu = self.job_dir.joinpath('neutra')
        for dir in [self.dir_log , self.dir_sku , self.dir_pqt , self.dir_res , self.dir_elt , self.dir_neu]:
            dir.mkdir(parents=True, exist_ok=True)

        self.paths : dict[str, Path] = {
            'elitelog' : self.job_dir.joinpath('elite_log.csv') ,
            'hoflog' : self.job_dir.joinpath('hof_log.csv') ,
            'runtime' : self.job_dir.joinpath('saved_times.csv') ,
            'params' : self.job_dir.joinpath('gp_params.pt') ,
            'df_axis' : self.job_dir.joinpath('df_axis.pt')
        }
        
        self.df_axis = {}

    def update_toolbox(self , toolbox : BaseToolbox):
        self.toolbox = toolbox
        return self

    def load_generation(self , i_iter = 0 , i_gen = 0 , hof_num = 500 , **kwargs):
        self.logbook = tools.Logbook()
        if i_iter < 0:
            pattern = r'iter(\d+)_.*\.pkl'
            matches = [re.match(pattern, file_name.name) for file_name in self.dir_log.iterdir()]
            i_iter = sorted(list(set([int(match.group(1)) for match in matches if match])))[i_iter]
            
        basename = self.record_basename(i_iter-1 , -1) if i_gen < 0 else self.record_basename(i_iter , i_gen)
            
        pop = []
        hof = tools.HallOfFame(hof_num)
        fbd = [] 
        if self.dir_log.joinpath(f'{basename}.pkl').exists():
            log = joblib.load(self.dir_log.joinpath(f'{basename}.pkl'))
            self.logbook.record(**log)
            fbd = [ind for ind in log['forbidden']] 
            if i_gen >= 0:
                # only update pop and hof when i_gen >= 0
                pop = [self.toolbox.to_ind(ind) for ind in log['population']] 
                hof_syx = [self.toolbox.to_syx(ind) for ind in log['halloffame']]
                hof_syx = self.toolbox.evaluate_pop(hof_syx, i_iter = i_iter, i_gen = i_gen, desc = 'Load HallofFame') # re-evaluate hof
                hof.update(hof_syx)
        elif i_gen >= 0:
            raise Exception(f'{self.dir_log.joinpath(f"{basename}.pkl")} does not exists!')
        else:
            log = None

        return pop , hof , fbd

    def dump_generation(self , population : Sequence , halloffame : Sequence | Any , forbidden : Sequence , i_iter = 0 , i_gen = 0 , **kwargs):
        if i_iter < 0: 
            return self
        basename = self.record_basename(i_iter , i_gen)

        # input type: population as getattr(creator , 'Individual') , halloffame as creator.Syntax , forbidden as creator.Syntax (most likely)
        # save type: syntax list (syntax_str , ind_str , fitness)
        self.logbook.record(i_gen = i_gen , 
                            population = [self.toolbox.to_record(ind) for ind in population], 
                            halloffame = [self.toolbox.to_record(ind) for ind in halloffame], 
                            forbidden = [self.toolbox.to_record(ind) for ind in forbidden], 
                            **kwargs)

        joblib.dump(self.logbook[-1] , self.dir_log.joinpath(f'{basename}.pkl'))
        return self

    def update_sku(self , individual , pool_skuname : str):
        poolid = int(pool_skuname.split('_')[-1])
        if poolid % 100 == 0:
            start_time_sku = datetime.now()
            output_path = self.dir_sku.joinpath(f'z_{pool_skuname}.txt')
            with open(output_path, 'w', encoding='utf-8') as file1:
                Logger.stdout(SyntaxControl.trim_syntax(individual),'\n start_time',str(start_time_sku),file=file1)

    def record_basename(self , i_iter = 0 , i_gen = 0):
        iter_str = 'iteration' if i_iter < 0 else f'iter{i_iter}'
        gen_str  = 'overall' if i_gen < 0 else f'gen{i_gen}'
        return f'{iter_str}_{gen_str}'
    
    def load_state(self , key , i_iter , i_gen = 0 , i_elite = 0 , device = None):
        if key in ['res' , 'neu' , 'elt']:
            dir : Path = getattr(self , f'dir_{key}')
            if i_iter < 0:
                pattern = r'iter(\d+).pt'
                matches = [re.match(pattern, file.name) for file in dir.iterdir()]
                i_iter = sorted(list(set([int(match.group(1)) for match in matches if match])))[i_iter]
            return torch_load(dir.joinpath(f'iter{i_iter}.pt')).to(device)
        elif key == 'parquet':
            return pd.read_parquet(self.dir_pqt.joinpath(f'elite_{i_elite}.parquet'), engine='fastparquet')
        else:
            path = self.paths[key]
            if key == 'df_axis':
                self.df_axis = torch_load(path)
                return self.df_axis
            elif key in ['elitelog' , 'hoflog']:
                df = pd.DataFrame()
                if path.exists():
                    df = pd.read_csv(path,index_col=0)
                    if i_iter >= 0: 
                        df = df[df.i_iter < i_iter]
                return df
            elif path.suffix == '.csv':
                return pd.read_csv(path,index_col=0)
            elif path.suffix == '.pt':
                return torch_load(path)
            else:
                raise Exception(key)

    def save_state(self , data , key , i_iter , i_gen = 0 , i_elite = 0 , **kwargs):
        if key in ['res' , 'neu' , 'elt']:
            dir = getattr(self , f'dir_{key}')
            torch.save(data , dir.joinpath(f'iter{i_iter}.pt'))
        elif key == 'parquet':
            assert isinstance(data , pd.DataFrame) , data
            data.to_parquet(self.dir_pqt.joinpath(f'elite_{i_elite}.parquet'),engine='fastparquet')
        else:
            path = self.paths[key]
            if key == 'df_axis':
                torch.save(data , self.paths['df_axis'])
                self.df_axis = data
            elif isinstance(data , pd.DataFrame) and path.suffix == '.csv':
                data.reset_index(drop=True).to_csv(path)
            elif path.suffix == '.pt':
                torch.save(data , path)
            else:
                raise Exception(key , data)
            
    def load_states(self, keys , **kwargs):
        return [self.load_state(key , **kwargs) for key in keys]
    
    def save_states(self, datas , **kwargs):
        return [self.save_state(data , key , **kwargs) for key , data in datas.items()]