import numpy as np
import pandas as pd
import torch

from typing import Callable , ClassVar , Iterator , Optional

from .base import BasicCallBack , WithCallBack
from ...classes import BatchData , BatchOutput , NdData
from ...environ import DIR

class CollectHidden(BasicCallBack):
    '''load booster data at fit end'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        self._print_info()
    @property
    def train_dl(self) -> Iterator[BatchData]: return self.data_mod.train_dataloader()
    @property
    def val_dl(self) -> Iterator[BatchData]: return self.data_mod.val_dataloader()
    @property
    def y_secid(self) -> np.ndarray | torch.Tensor: return self.data_mod.y_secid
    @property
    def y_date(self) -> np.ndarray | torch.Tensor: return self.data_mod.y_date
    def hidden_3d(self , dataloader : Iterator[BatchData]) -> Optional[NdData]:
        hh , ii = [] , []
        for batch_data in dataloader:
            hidden = BatchOutput(self.module.net(batch_data.x)).hidden
            if hidden is None: return
            hh.append(hidden.detach().cpu().numpy())
            ii.append(batch_data.i)
    
        hh , ii = np.vstack(hh) , np.vstack(ii)
        secid_i , secid_j = np.unique(ii[:,0] , return_inverse=True)
        date_i  , date_j  = np.unique(ii[:,1] , return_inverse=True)
        values = np.full((len(secid_i) , len(date_i) , hh.shape[-1]) , fill_value=np.nan)
        values[secid_j , date_j] = hh[:]
        index = [self.y_secid[secid_i] , self.y_date[date_i] , np.array([f'Hidden{i}' for i in range(hh.shape[-1])])]

        print(values.shape)
        print(f'Finite Ratio : {np.isfinite(values).sum() / np.prod(values.shape) :.4f}')
        print(index)
        return NdData(values , index)

    def on_fit_model_start(self):
        self.module.net.eval()
        with torch.no_grad():
            train_hidden = self.hidden_3d(self.train_dl)
            valid_hidden = self.hidden_3d(self.val_dl)
        print(train_hidden)
        return


            