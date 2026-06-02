"""Data containers for the boost pipeline.

Classes:
    BoostOutput      — flat prediction container with secid/date index
    BoostInput       — aligned 3-D tensor container with weight computation
    BoostWeightMethod — three-axis (ts/cs/bm) sample weight calculator
"""
from __future__ import annotations
import torch
import numpy as np
import pandas as pd
import xarray as xr

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any , Literal

from src.proj import Logger , BaseProperty
from src.func import index_merge , match_slice , intersect_meshgrid
from src.func.metric import rankic_2d , ic_2d
from src.func.tensor import rank_pct

from .weight import BoostWeightMethod

__all__ = ['BoostDataset' , 'BoostOutput' , 'BoostInput']

@dataclass(slots=True)
class BoostDataset:
    x : torch.Tensor
    y : torch.Tensor | None
    w : torch.Tensor
    date : np.ndarray

    def __post_init__(self):
        self.x = self.x.cpu()
        self.y = self.y.cpu() if self.y is not None else None
        self.w = self.w.cpu()

    @property
    def shape(self):
        return BaseProperty.shape({'x' : self.x , 'y' : self.y , 'w' : self.w , 'date' : self.date})

    def to(self , device : torch.device | None = None):
        self.x = self.x.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        self.w = self.w.to(device)
        return self
    
    def boost_inputs(
        self , boost_type : Literal['lgbm' , 'xgboost' , 'catboost'] , rank = False
    ) -> dict[str,Any]:
        label = self.y.detach().cpu().numpy() if self.y is not None else None
        data : dict[str,Any] = {'data' : self.x.detach().cpu().numpy()}
        if rank:
            if boost_type == 'lgbm':
                # One query per trade date; required for lambdarank/NDCG and correct cross-section semantics.
                data['group'] = self.group_arr()
                if label is not None:
                    data['label'] = label.astype(np.int32)
            elif boost_type == 'xgboost':
                data['group'] = self.group_arr()
                data['weight'] = self.group_weight()
                if label is not None:
                    data['label'] = (label / 5).astype(np.int32)
            elif boost_type == 'catboost':
                if label is not None:
                    data['label'] = label.astype(np.int32)
                data['group_id'] = self.group_id()
                data['group_weight'] = self.broadcasted_group_weights()
            else:
                raise ValueError(f'Boost type {boost_type} not supported')
        else:
            data['label'] = label
            data['weight'] = self.w.detach().cpu().numpy()
        return data

    def to_lgbm_dataset(self , rank=False , reference=None):
        import lightgbm
        data = self.boost_inputs('lgbm', rank=rank)
        return lightgbm.Dataset(**data , reference=reference)

    def to_xgboost_dataset(self , rank=False):
        import xgboost
        data = self.boost_inputs('xgboost' , rank=rank)
        return xgboost.DMatrix(**data)

    def to_catboost_dataset(self , rank=False):
        import catboost
        data = self.boost_inputs('catboost' , rank=rank)
        return catboost.Pool(**data)
    
    def group_arr(self):
        assert (np.diff(self.date) >= 0).all() , 'date must be sorted'
        return np.unique(self.date , return_counts=True)[1]

    def group_weight(self) -> np.ndarray:
        """Per-date sum of sample weights (length ``n_groups``).

        Uses ``np.add.reduceat`` on contiguous date blocks (same ``date`` sort
        invariant as :meth:`group_arr`). Pass ``w`` from :meth:`as_numpy` to
        avoid an extra tensor copy when building boost inputs.
        """
        counts = self.group_arr()
        if counts.size == 0:
            return np.array([], dtype=float)
        starts = np.concatenate(([0], counts.cumsum()[:-1]))
        return np.add.reduceat(self.w.detach().cpu().numpy(), starts)

    def broadcasted_group_weights(self) -> np.ndarray:
        """Per-row group weights for CatBoost (length ``n_samples``).

        Pool ``group_weight`` must match ``len(data)``; each row gets its date's
        aggregated weight from :meth:`group_weight` via ``np.repeat``.
        """
        counts = self.group_arr()
        if counts.size == 0:
            return np.array([], dtype=np.float64)
        return np.repeat(self.group_weight(), counts)
    
    def group_id(self): 
        return self.date

    @property
    def nfeat(self): 
        return self.x.shape[-1]

@dataclass
class BoostOutput:
    """Container for boost model predictions.

    Attributes:
        pred:   Flat prediction tensor for the finite (non-NaN) positions.
        secid:  Security IDs array of shape ``(n_sample,)``.
        date:   Date array of shape ``(n_date,)``.
        finite: Boolean mask of shape ``(n_sample, n_date)`` indicating
                non-NaN positions.  Used to map flat ``pred`` back to the
                full 2-D grid.
        label:  Optional ground-truth tensor of shape ``(n_sample, n_date)``
                for IC evaluation.
    """
    pred    : torch.Tensor
    secid   : np.ndarray
    date    : np.ndarray
    finite  : torch.Tensor
    label   : torch.Tensor | Any = None
    input   : BoostInput | None = None

    def __post_init__(self):
        assert len(self.pred) == len(self.secid) or len(self.pred) == self.finite.sum(), (len(self.pred) , len(self.secid) , self.finite.sum())
        self._raw_pred = self.pred.clone().cpu()
        if len(self.pred) != len(self.secid):
            pred = torch.full((len(self.secid),len(self.date)), fill_value=np.nan).to(self.pred)
            pred[self.finite] = self.pred
            self.pred = pred
        elif self.pred.ndim == 1:
            self.pred = self.pred.unsqueeze(1)
        assert self.pred.shape == (len(self.secid),len(self.date)), (self.pred.shape , len(self.secid) , len(self.date))

    @property
    def shape(self):
        return BaseProperty.shape({'pred' : self.pred , 'secid' : self.secid , 'date' : self.date , 'finite' : self.finite , 'label' : self.label})

    def to_dataframe(self):
        df = pd.DataFrame(self.pred.cpu().numpy() , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df
    
    def rankic(self):
        assert self.label is not None , f'{self.__class__.__name__} label is None'
        return rankic_2d(self.pred , self.label , 0)

    def top5pct(self):
        assert self.label is not None , f'{self.__class__.__name__} label is None'
        top_pos = rank_pct(self.pred) >= 0.95
        return torch.where(top_pos , self.label , torch.nan).nanmean(0)
    
    def ic(self):
        assert self.label is not None , f'{self.__class__.__name__} label is None'
        return ic_2d(self.pred , self.label , 0)

class BoostInput:
    """Aligned 3-D tensor container for boost model input.

    Attributes:
        x:            Feature tensor of shape ``(n_sample, n_date, n_feature)``.
        y:            Label tensor of shape ``(n_sample, n_date)``, or ``None``.
        w:            Pre-computed sample weight tensor of same shape as ``y``.
                      When ``None``, weights are derived on-the-fly from
                      ``weight_method``.
        secid:        Security ID array of shape ``(n_sample,)``.
        date:         Date array of shape ``(n_date,)``.
        feature:      Feature name array of shape ``(n_feature,)``.
        weight_param: Keyword arguments forwarded to :class:`BoostWeightMethod`.
                      Supported keys:
                        * ``ts_type``           : ``'lin'`` | ``'exp'`` | ``None``
                        * ``cs_type``           : ``'top'`` | ``'positive'`` | ``'ones'`` | ``None``
                        * ``bm_type``           : ``'in'`` | ``None``
                        * ``ts_lin_rate``       : float (default 0.5)
                        * ``ts_half_life_rate`` : float (default 0.5)
                        * ``cs_top_tau``        : float
                        * ``cs_ones_rate``      : float (default 2.0)
                        * ``bm_rate``           : float (default 2.0)
                        * ``bm_secid``          : benchmark security IDs or ``None``
        n_bins:       When not ``None``, ``y`` is converted to integer category
                      labels in ``[0, n_bins - 1]`` for classification training.
    """
   
    def __init__(
        self , x : torch.Tensor , y : torch.Tensor | None = None , w : torch.Tensor | None = None ,
        secid : np.ndarray | None = None , date : np.ndarray | None = None , feature : np.ndarray | list[str] | None = None ,
        weight_param : dict[str,Any] | None = None , n_bins : int | None = None , use_feature : np.ndarray | list[str] | None = None):
        self._x = x
        self._y = y
        self._w = w
        self._secid = secid
        self._date = date
        self._feature = feature
        self._weight_param = weight_param or {}
        self._n_bins = n_bins
        self._use_feature = use_feature

    def __repr__(self):
        return '\n'.join(
            [f'secid={self.secid}', 
             f'date={self.date}', 
             f'feature={self.feature}', 
             f'weight_method={self.weight_method})'])

    @property
    def x(self) -> torch.Tensor:
        return self._x
    @property
    def y(self) -> torch.Tensor | None:
        return self._y
    @property
    def w(self) -> torch.Tensor:
        if self._w is None:
            if self.y is None:
                return torch.ones_like(self.x[:,:,0])
            else:
                return self.weight_method.calculate_weight(self.y , self.secid) 
        else:
            return self._w
    @property
    def secid(self) -> np.ndarray:
        if self._secid is None:
            return np.arange(self.x.shape[0])
        else:
            return self._secid
    @property
    def date(self) -> np.ndarray:
        if self._date is None:
            return np.arange(self.x.shape[1])
        else:
            return self._date
    @property
    def feature(self) -> np.ndarray:
        if self._feature is None:
            return np.array([f'F.{i}' for i in range(self.x.shape[-1])])
        else:
            return np.array(self._feature)

    @property
    def weight_param(self) -> dict[str,Any]:
        return self._weight_param

    @property
    def n_bins(self) -> int | None:
        return self._n_bins

    @property
    def use_feature(self) -> np.ndarray | None:
        return np.array(self._use_feature) if self._use_feature is not None else None

    @property
    def feat_idx(self): 
        if self._use_feature is None:
            return slice(None)
        else:
            assert all(np.isin(self._use_feature , self.feature)) , np.setdiff1d(self._use_feature , self.feature)
            return match_slice(self._use_feature , self.feature)

    @property
    def shape(self): 
        return self.x.shape

    @property
    def nfeat(self): 
        return len(self.feature) if self._use_feature is None else len(self._use_feature)

    @property
    def complete(self) -> bool:
        return self.y is not None and self.secid is not None and self.date is not None and self.feature is not None

    @property
    def is_categorical(self) -> bool: 
        return self.n_bins is not None

    @cached_property
    def weight_method(self) -> BoostWeightMethod:
        return BoostWeightMethod(**self.weight_param)

    def copy(self) -> BoostInput:
        return deepcopy(self)

    def set_data_param(self , use_feature : np.ndarray | list[str] | None = None , n_bins : int | None = None , weight_param : dict[str,Any] | None = None): 
        if use_feature is not None and (self._use_feature is None or not np.array_equal(use_feature , self._use_feature)):
            self._use_feature = np.array(use_feature)
            self.__dict__.pop('X' , None)

        if n_bins is not None and n_bins != self._n_bins:
            self._n_bins = n_bins
            self.__dict__.pop('Y' , None)

        if weight_param is not None:
            self.weight_method.reset(**weight_param)
            self.__dict__.pop('W' , None)
        return self

    @cached_property
    def finite(self) -> torch.Tensor:
        finite = ~self.x.isnan().all(dim=-1)
        if self.y is not None:
            finite = finite & (self.y >= 0 if self.is_categorical else self.y.isfinite())
        return finite

    @cached_property
    def finite_idx(self) -> tuple[torch.Tensor, torch.Tensor]:
        tr_finite = self.finite.transpose(0,1)
        idx0 = torch.arange(self.x.shape[0])[None,:].where(tr_finite , torch.nan)[tr_finite].to(torch.int)
        idx1 = torch.arange(self.x.shape[1])[:,None].where(tr_finite , torch.nan)[tr_finite].to(torch.int)
        return idx0 , idx1

    def _flatten_by_date(self , obj : torch.Tensor | None) -> torch.Tensor | Any:
        """Flatten x/y/w by the finite mask.

        Args:
            obj: Tensor aligned with ``(n_sample, n_date, ...)``.

        Returns:
            Flattened array of length ``n_finite``
        """
        if obj is None:
            return obj
        idx0 , idx1 = self.finite_idx
        return obj[idx0 , idx1]

    @cached_property
    def SECID(self) -> np.ndarray:
        """Return flat security-ID array aligned with the finite mask."""
        return self.secid[self.finite_idx[0]]

    @cached_property
    def DATE(self) -> np.ndarray:
        """Return flat date array aligned with the finite mask."""
        return self.date[self.finite_idx[1]]

    @cached_property
    def X(self) -> torch.Tensor:
        """Return flat feature matrix of shape ``(n_finite, n_use_feature)``.

        Only the columns in ``use_feature`` are returned; NaN rows are dropped.
        ! important: rank_pct is applied to every date every feature
        """
        rank_x = rank_pct(self.x[...,self.feat_idx] , dim = 0).clip(0 , 0.9999) * 100
        return self._flatten_by_date(rank_x)

    @cached_property
    def Y(self) -> torch.Tensor | None:
        """
        Return flat label vector of length ``n_finite``, NaN rows dropped.
        ! important: rank_pct is applied to every date
        """
        if self.y is None:
            return None
        else:
            rank_y = rank_pct(self.y , dim = 0).clip(0 , 0.9999)
            if self.n_bins is None:
                rank_y = rank_y * 100
            else:
                rank_y = (rank_y * self.n_bins).nan_to_num(-1).int().clip(-1 , self.n_bins-1)
            return self._flatten_by_date(rank_y)
        
    @cached_property
    def W(self) -> torch.Tensor:
        """Return flat sample weight vector of length ``n_finite``, NaN rows dropped.

        If ``self.w`` is ``None`` the weights are computed on-the-fly from
        ``weight_method.calculate_weight()``.
        """
        return self._flatten_by_date(self.w)
    
    def Dataset(self): 
        return BoostDataset(self.X , self.Y , self.W , self.DATE)
        
    def output(self , pred : torch.Tensor | np.ndarray | Any) -> BoostOutput:
        """Wrap a flat prediction array into a :class:`BoostOutput`.

        Parameters
        ----------
        pred: Flat prediction array of length ``n_finite``.
              If 2-D (softmax output with shape ``(n_finite, n_class)``),
              a linear combination with centred class indices is applied to
              collapse to a scalar score.

        Returns
        -------
        :class:`BoostOutput` with ``secid``, ``date``, and ``finite`` from this
        container and ``label`` set to the original continuous ``_raw_y``.
        """
        if isinstance(pred , torch.Tensor):
            new_pred = pred
        else:
            new_pred = torch.from_numpy(np.array(pred , copy=True , dtype=np.float32))

        if new_pred.ndim == 2: 
            weight = torch.arange(new_pred.shape[1]).to(new_pred) - (new_pred.shape[1] - 1) / 2
            new_pred = new_pred @ weight
        elif new_pred.ndim > 2:
            raise ValueError(f'BoostOutput cannot deal with pred with ndim {new_pred.ndim}')
        return BoostOutput(new_pred , self.secid , self.date , self.finite , self.y , self)

    def pred_to_dataframe(self , pred : np.ndarray | torch.Tensor):
        new_pred = pred.numpy() if isinstance(pred , torch.Tensor) else pred 
        df = pd.DataFrame(new_pred , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df

    def to_dataframe(self):
        df = pd.DataFrame(self.X.cpu().numpy() , columns = self.feature)
        df['secid'] = self.SECID
        df['date']  = self.DATE
        df['label'] = self.Y.cpu().numpy() if self.Y is not None else np.nan
        df = df.set_index(['secid' , 'date'])
        return df

    @classmethod
    def from_dataframe(cls , data : pd.DataFrame , weight_param : dict[str,Any] | None = None , label_col : str | None = None):
        """Construct from a tidy ``DataFrame`` with a secid/date multi-index.

        The label column is treated as the label (``y``); all other columns
        become features.  The index is auto-detected from common column names:
        ``['SecID','instrument','secid','StockID']`` and
        ``['TradeDate','datetime','date']``.  The label column is auto-detected
        from common column names: ``['label']``.
        """
        weight_param = weight_param or {}
        SECID_COLS = ['SecID','instrument','secid','StockID']
        DATE_COLS  = ['TradeDate','datetime','date']  
        if data.index.name or len(data.index.names) > 1: 
            data = data.reset_index()

        var_sec  = np.intersect1d(SECID_COLS , data.columns.to_numpy())
        var_date = np.intersect1d(DATE_COLS  , data.columns.to_numpy())
        assert len(var_sec) == len(var_date) == 1, (var_sec , var_date , data.columns)
        data = data.rename(columns={var_sec[0]:'secid' , var_date[0]:'date'}).set_index(['secid' , 'date'])

        if label_col is None:
            label_col = data.columns.to_list()[-1]
        if not label_col.lower().startswith(('ret' , 'y' , 'label' , 'rtn' , 'res' , 'std')):
            Logger.warning(f'using {label_col} as label column, not recommended')
        xarr = xr.Dataset.from_dataframe(data.drop(columns=[label_col]))
        yarr = xr.Dataset.from_dataframe(data[[label_col]])

        xindex = [arr.values for arr in xarr.indexes.values()] + [list(xarr.data_vars)]
        x = torch.Tensor(np.stack([arr.to_numpy() for arr in xarr.data_vars.values()] , -1))
        y = torch.Tensor(np.stack([arr.to_numpy() for arr in yarr.data_vars.values()] , -1)[...,0])
        
        secid , date , feature = xindex[0] , xindex[1] , xindex[-1]
        return cls(x , y , None , secid , date , feature , weight_param)

    @classmethod
    def from_numpy(
        cls , x : np.ndarray , y : np.ndarray | None = None,  w : np.ndarray | None = None ,
        secid : Any = None , date : Any = None , feature : Any = None ,
        weight_param : dict[str,Any] | None = None):
        """Construct from NumPy arrays, delegating to :meth:`from_tensor`."""
        return cls.from_tensor(
            x = torch.Tensor(x) , 
            y = None if y is None else torch.Tensor(y) ,
            w = None if w is None else torch.Tensor(w) ,
            secid = secid , date = date , feature = feature , weight_param = weight_param)
    
    @classmethod
    def from_tensor(
        cls , x : torch.Tensor , y : torch.Tensor | None = None , w : torch.Tensor | None = None ,
        secid : Any = None , date : Any = None , feature : Any = None ,
        weight_param : dict[str,Any] | None = None):
        """Construct from tensors, normalising shapes and generating default indices.

        ``x`` may be 2-D ``(n_sample, n_feature)`` (treated as single date) or
        3-D ``(n_sample, n_date, n_feature)``.  ``y`` may be 1-D, 2-D, or 3-D
        with a trailing size-1 feature axis which is squeezed.

        Default ``secid``/``date``/``feature`` are integer ``arange`` sequences
        when not provided.
        """
        assert x.ndim in [2,3] , x.ndim
        assert y is None or y.ndim in [x.ndim - 1, x.ndim] , (y.ndim , x.ndim)
        if y is not None and y.ndim == x.ndim:
            assert y.shape[-1] == 1 , f'Boost Data cannot deal with multilabels, but got {y.shape}'
            y = y[...,0]
        if y is not None and y.ndim == 1: 
            y = y[:,None]
        if x.ndim == 2:  
            x = x[:,None,:]

        if secid is None:  
            secid = np.arange(x.shape[0])
        if date  is None : 
            date  = np.arange(x.shape[1])
        if feature is None : 
            feature = np.array([f'F.{i}' for i in range(x.shape[-1])])
        return cls(x , y , w , secid , date , feature , weight_param)
    
    @classmethod
    def concat(cls , datas : list[BoostInput | None]) -> BoostInput:
        """Union-merge a list of :class:`BoostInput` objects along all axes.

        ``secid`` and ``date`` are union-merged; ``feature`` is stacked
        (concatenated without deduplication).  Overlapping ``x``/``y`` cells
        from later blocks overwrite earlier ones.  ``None`` entries and
        incomplete blocks (``complete == False``) are silently skipped.
        """
        blocks = [data for data in datas if data is not None and data.complete]
        
        secid   = index_merge([blk.secid   for blk in blocks] , method = 'union')
        date    = index_merge([blk.date    for blk in blocks] , method = 'union')
        feature = index_merge([blk.feature for blk in blocks] , method = 'stack')
        
        x = torch.full((len(secid) , len(date) , len(feature)) , fill_value=torch.nan)
        y = torch.full((len(secid) , len(date)) , fill_value=torch.nan)
        w = None if all(blk.w is None for blk in blocks) else torch.ones_like(y)
        for i , blk in enumerate(blocks): 
            tar_grid , src_grid = intersect_meshgrid([secid , date , feature] , [blk.secid , blk.date , blk.feature])
            x[*tar_grid] = blk.x[*src_grid]

            tar_grid , src_grid = intersect_meshgrid([secid , date] , [blk.secid , blk.date])
            if blk.y is not None:
                y[*tar_grid] = blk.y[*src_grid]
            if blk.w is not None and w is not None:
                w[*tar_grid] = blk.w[*src_grid]
        if y.isnan().all():
            y = None
        new_binput = cls(x , y , w , secid , date , feature)
        return new_binput