import gc
import pandas as pd
import torch

from tqdm import tqdm

from src.proj import Logger
from src.res.deap.func import math_func as MF , factor_func as FF
from .memory import MemoryManager

class gpEliteGroup:
    def __init__(self , start_i_elite = 0 , device = None , block_len = 50) -> None:
        self.start_i_elite = start_i_elite
        self.i_elite = start_i_elite
        self.device  = device
        self.block_len = block_len
        self.container = []
        self.position  = []

    def assign_logs(self , hof_log , elite_log):
        self.hof_log = hof_log
        self.elite_log = elite_log
        return self

    def update_logs(self , new_log):
        if len(self.elite_log):
            self.elite_log = pd.concat([self.elite_log , new_log[new_log.elite]] , axis=0) 
        else:
            self.elite_log = new_log[new_log.elite]
        self.hof_log = pd.concat([self.hof_log , new_log] , axis=0) if len(self.hof_log) else new_log
        return self

    def max_corr_with_me(self , factor , abs_corr_cap = 1.01 , dim = 1 , dim_valids = (None , None) , syntax = None):
        assert isinstance(factor , FF.FactorValue) , type(factor)
        corr_values = torch.zeros((self.i_elite - self.start_i_elite + 1 ,))
        if isinstance(factor.value , torch.Tensor): 
            corr_values = corr_values.to(factor.value)
        exit_state  = False
        idx = 0
        for block in self.container:
            corrs , exit_state = block.max_corr(factor , abs_corr_cap , dim , dim_valids , syntax = syntax)
            corr_values[idx:idx+block.len()] = corrs[:block.len()]
            idx += block.len()
            if exit_state: 
                break
        return corr_values , exit_state

    def append(self , factor , starter = None , **kwargs):
        assert isinstance(factor , FF.FactorValue) , type(factor)
        if len(self.container) and not self.container[-1].full:
            self.container[-1].append(factor , **kwargs)
        else:
            if len(self.container): 
                self.container[-1].cat2cpu()
            self.container.append(gpEliteBlock(self.block_len).append(factor , **kwargs))
        self.position.append((len(self.container)-1,self.container[-1].len()-1))
        if isinstance(starter,str): 
            Logger.stdout(f'{starter}Elite{self.i_elite:_>3d} (' + '|'.join([f'{k}{v:+.2f}' for k,v in kwargs.items()]) + f'): {factor.name}')
        self.i_elite += 1
        return self
    
    def cat_all(self):
        for block in self.container:
            block.cat2cpu()
        return self
    
    def compile_elite_tensor(self , device = None):
        if self.container:
            self.cat_all()
            if device is None: 
                device = self.device
            torch.cuda.empty_cache()
            try:
                new_tensor = torch.cat([block.data_to_device(device) for block in self.container] , dim = -1)
            except torch.cuda.OutOfMemoryError:
                Logger.warning('OutofMemory when compiling elite tensor, try use cpu to concat')
                new_tensor = torch.cat([block.data_to_device('cpu') for block in self.container] , dim = -1)
                new_tensor = new_tensor.to(device)
        else:
            new_tensor = None
        return new_tensor
    
    def total_len(self):
        return sum([blk.len() for blk in self.container])
    
    def total_mem(self):
        return sum([MemoryManager.object_memory(blk) for blk in self.container])

    def select(self , i):
        return self.container[self.position[i][0]].select(self.position[i][1])

    def corrmat_of_all(self):
        '''
        Calculate correlation matrix of all members
        The corrmat is average(over dim 0) corr(over dim 1), of shape nfactors * nfactors(dim 2)
        '''
        total = self.total_len()
        corr_mat = torch.eye(total).to(self.device)
        iterator = [(i,*self.position[i],j,*self.position[j]) for i in range(total) for j in range(i+1,total)]
        iter_df = pd.DataFrame(iterator , columns = pd.Index(['ii','ib','ik','jj','jb','jk']))
        iter_df = iter_df.sort_values(['ib' , 'jb' , 'ik' , 'jk'])
        Logger.stdout(f'Total Correlation Counts : {len(iter_df)}')
        for grp , sub_df in iter_df.groupby(['ib' , 'jb']):
            ib , jb = int(grp[0]) , int(grp[1]) # type: ignore
            Blk_i = self.container[ib].data_to_device(self.device)
            Blk_j = self.container[jb].data_to_device(self.device)
            for _ , (ii,ik,jj,jk) in tqdm(sub_df.loc[:,['ii','ik','jj','jk']].iterrows(),
                                        desc=f'Block_{ib}/Block_{jb}',total=len(sub_df)):
                corr = MF.corrwith(Blk_i[...,ik] , Blk_j[...,jk] , dim=-1)
                corr_mat[ii, jj] = corr.nanmean() if isinstance(corr , torch.Tensor) else torch.nan

        corr_mat = torch.where(corr_mat == 0 , corr_mat.T , corr_mat).cpu()
        return corr_mat
    
class gpEliteBlock:
    def __init__(self , max_len = 50):
        self.max_len = max_len
        self.names = []
        self.infos = {}
        self.data  = []
        self.full = False

    def len(self):
        return len(self.names)
        
    def cat2cpu(self):
        if isinstance(self.data , list): 
            try:
                data = MF.concat_factors(*self.data)
                if isinstance(data , torch.Tensor): 
                    self.data = data.cpu()
            except MemoryError:
                Logger.warning('OutofMemory when concat gpEliteBlock, try use cpu to concat')
                gc.collect()
                self.data = MF.concat_factors(*self.data , device=torch.device('cpu')) # to cpu first
        assert self.data is not None , f'{self} has data None'
        return self

    def append(self , factor , **kwargs):
        assert isinstance(factor , FF.FactorValue) , type(factor)
        if not self.full and isinstance(self.data , list):
            self.names.append(factor.name)
            self.infos.update({factor.name:kwargs})
            self.data.append(factor.value)
            self.full = self.len() >= self.max_len
        else:
            raise Exception('The EliteBlock is Full')   
        return self
    
    def max_corr(self , factor , abs_corr_cap = 1.01 , dim = None , dim_valids = (None , None) , syntax = None):
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
            corr = MF.corrwith(value, blk , dim=dim).nanmean() 
            corr_values[k] = corr
            if exit_state := corr.abs() > abs_corr_cap: 
                break 
        return corr_values , exit_state
    
    def data_to_device(self , device , inplace = False):
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