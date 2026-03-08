import gc
import pandas as pd
import torch
from typing import Any
from tqdm import tqdm

from src.proj import Logger
from src.func import tensor as T
from src.res.gp.func import factor_func as FF
from .memory import MemoryManager

class EliteGroup:
    def __init__(self , start_i_elite = 0 , device = None , block_max_len = 50) -> None:
        self.start_i_elite = start_i_elite
        self.elite_count   = 0
        self.device    = device
        self.block_max_len = block_max_len
        self.blocks : list[EliteBlock] = []
        self.elite_names : list[str] = []
        self.elite_positions : list[tuple[int,int]] = []

    def __repr__(self):
        return f'EliteGroup(num_elites={self.total_len()})'

    @property
    def i_elite(self):
        return self.elite_count + self.start_i_elite

    @classmethod
    def new_from_logs(cls , elite_log : pd.DataFrame , hof_log : pd.DataFrame | None = None , * , device = None):
        elite_group = cls(start_i_elite = len(elite_log) , device = device)
        elite_group.assign_logs(elite_log , hof_log)
        return elite_group

    def assign_logs(self , elite_log : pd.DataFrame , hof_log : pd.DataFrame | None = None):
        self.elitelog = elite_log
        self.hoflog = hof_log
        return self

    def update_logs(self , new_log : pd.DataFrame):
        if self.elitelog is not None and not self.elitelog.empty:
            self.elitelog = pd.concat([self.elitelog , new_log[new_log['elite']]] , axis=0) 
        else:
            self.elitelog = new_log[new_log.elite]
        self.hoflog = pd.concat([self.hoflog , new_log] , axis=0) if self.hoflog is not None and not self.hoflog.empty else new_log
        return self

    def max_corr_with_me(self , factor : FF.FactorValue , abs_corr_cap = 1.01 , dim = 1 , dim_valids = (None , None) , syntax = None):
        assert isinstance(factor , FF.FactorValue) , type(factor)
        corr_values = torch.zeros((self.i_elite - self.start_i_elite + 1 ,))
        if isinstance(factor.value , torch.Tensor): 
            corr_values = corr_values.to(factor.value)
        exit_state  = False
        idx = 0
        for block in self.blocks:
            corrs , exit_state = block.max_corr(factor , abs_corr_cap , dim , dim_valids , syntax = syntax)
            corr_values[idx:idx+block.len()] = corrs[:block.len()]
            idx += block.len()
            if exit_state: 
                break
        return corr_values , exit_state

    def append(self , factor : FF.FactorValue , **kwargs):
        assert factor.name not in self.elite_names , f'Elite {factor.name} already exists'
        if len(self.blocks) and not self.blocks[-1].full:
            self.blocks[-1].append(factor , **kwargs)
        else:
            if len(self.blocks): 
                self.blocks[-1].cat2cpu()
            self.blocks.append(EliteBlock(self.block_max_len).append(factor , **kwargs))
        self.elite_names.append(factor.name)
        self.elite_positions.append((len(self.blocks)-1,self.blocks[-1].len()-1))
        self.elite_count += 1
        return self
    
    def cat_all(self):
        for block in self.blocks:
            block.cat2cpu()
        return self
    
    def compile_elite_tensor(self , device = None):
        if self.blocks:
            self.cat_all()
            if device is None: 
                device = self.device
            torch.cuda.empty_cache()
            try:
                new_tensor = torch.concat([block.data_to_device(device) for block in self.blocks] , dim = -1)
            except torch.cuda.OutOfMemoryError:
                Logger.warning('OutofMemory when compiling elite tensor, try use cpu to concat')
                new_tensor = torch.concat([block.data_to_device('cpu') for block in self.blocks] , dim = -1)
                new_tensor = new_tensor.to(device)
        else:
            new_tensor = None
        return new_tensor
    
    def total_len(self):
        return sum([blk.len() for blk in self.blocks])

    def all_names(self):
        return [name for blk in self.blocks for name in blk.names]
    
    def total_mem(self):
        return sum([MemoryManager.object_memory(blk) for blk in self.blocks])

    def select(self , elite : int | str):
        if isinstance(elite , str):
            elite = self.elite_names.index(elite)
        i , j = self.elite_positions[elite]
        return self.blocks[i].select(j)

    def corrmat_of_all(self):
        '''
        Calculate correlation matrix of all members
        The corrmat is average(over dim 0) corr(over dim 1), of shape nfactors * nfactors(dim 2)
        '''
        total = self.total_len()
        corr_mat = torch.eye(total).to(self.device)
        iterator = [(i,*self.elite_positions[i],j,*self.elite_positions[j]) for i in range(total) for j in range(i+1,total)]
        iter_df = pd.DataFrame(iterator , columns = pd.Index(['ii','ib','ik','jj','jb','jk']))
        iter_df = iter_df.sort_values(['ib' , 'jb' , 'ik' , 'jk'])
        Logger.stdout(f'Total Correlation Counts : {len(iter_df)}')
        for grp , sub_df in iter_df.groupby(['ib' , 'jb']):
            ib , jb = int(grp[0]) , int(grp[1]) # type: ignore
            Blk_i = self.blocks[ib].data_to_device(self.device)
            Blk_j = self.blocks[jb].data_to_device(self.device)
            for _ , (ii,ik,jj,jk) in tqdm(sub_df.loc[:,['ii','ik','jj','jk']].iterrows(),
                                        desc=f'Block_{ib}/Block_{jb}',total=len(sub_df)):
                corr = T.corrwith(Blk_i[...,ik] , Blk_j[...,jk] , dim=-1)
                corr_mat[ii, jj] = corr.nanmean() if isinstance(corr , torch.Tensor) else torch.nan

        corr_mat = torch.where(corr_mat == 0 , corr_mat.T , corr_mat).cpu()
        return corr_mat
    
class EliteBlock:
    def __init__(self , max_len = 50):
        self.max_len = max_len
        self.names : list[str] = []
        self.infos : dict[str,Any] = {}
        self.data  : list[torch.Tensor] | torch.Tensor | None = []

    def __repr__(self):
        return f'EliteBlock(num_elites={self.len()})'

    @property
    def full(self):
        return len(self.names) >= self.max_len

    def len(self):
        return len(self.names)
        
    def cat2cpu(self):
        if isinstance(self.data , list): 
            try:
                data = T.concat_factors_2d(*self.data)
                if isinstance(data , torch.Tensor): 
                    self.data = data.cpu()
            except MemoryError:
                Logger.warning('OutofMemory when concat gpEliteBlock, try use cpu to concat')
                gc.collect()
                self.data = T.concat_factors_2d(*self.data , device=torch.device('cpu')) # to cpu first
        assert self.data is not None , f'{self} has data None'
        assert isinstance(self.data , torch.Tensor) , type(self.data)
        assert self.data.dim() == 3 , f'{self} data dim must be 3, but got {self.data.dim()}'
        return self

    def append(self , factor , **kwargs):
        assert isinstance(factor , FF.FactorValue) , type(factor)
        if not self.full and isinstance(self.data , list):
            assert isinstance(factor.value , torch.Tensor) , type(factor.value)
            self.names.append(factor.name)
            self.infos.update({factor.name:kwargs})
            self.data.append(factor.value)
        else:
            raise Exception('The EliteBlock is Full')   
        return self
    
    def max_corr(self , factor : FF.FactorValue | torch.Tensor | Any , abs_corr_cap = 1.01 , dim = None , dim_valids = (None , None) , syntax = None):
        assert self.data is not None , f'{self} has data None'
        
        if isinstance(factor , FF.FactorValue): 
            factor = factor.value
        assert isinstance(factor , torch.Tensor) , factor
        corr_values = torch.zeros((self.len()+1,)).to(factor)
        exit_state  = False
        block = self.data.to(factor) if isinstance(self.data , torch.Tensor) else self.data
        i = torch.arange(factor.shape[0]) if dim_valids[0] is None else dim_valids[0]
        j = torch.arange(factor.shape[1]) if dim_valids[1] is None else dim_valids[1]
        value = factor[i][:,j]
        for k in range(self.len()):
            blk = block[i][:,j][...,k] if isinstance(block , torch.Tensor) else block[k][i][:,j]
            corr = T.corrwith(value, blk , dim=dim).nanmean() 
            corr_values[k] = corr
            if exit_state := corr.abs() > abs_corr_cap: 
                break 
        return corr_values , exit_state
    
    def data_to_device(self , device : torch.device | str | None , inplace = False):
        assert isinstance(self.data , torch.Tensor) , type(self.data)
        if device == 'cpu':
            data = self.data.cpu()
        else:
            data = self.data.to(device)
        if inplace: 
            self.data = data
        return data
    
    def select(self , i):
        assert self.data is not None , f'{self} has data None'
        return self.data[...,i] if isinstance(self.data , torch.Tensor) else self.data[i]