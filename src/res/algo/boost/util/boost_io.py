"""Data containers for the boost pipeline.

Classes:
    BoostOutput      — flat prediction container with secid/date index
    BoostInput       — aligned 3-D tensor container with weight computation
    BoostWeightMethod — three-axis (ts/cs/bm) sample weight calculator
"""
import torch
import numpy as np
import pandas as pd
import xarray as xr

from copy import deepcopy
from dataclasses import dataclass , field
from typing import Any , Literal

from src.func import match_values , index_merge , match_slice , intersect_meshgrid
from src.func.metric import rankic_2d , ic_2d
from src.proj import Logger

__all__ = ['BoostOutput' , 'BoostInput' , 'BoostWeightMethod']

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

    def to_2d(self):
        """Reconstruct the full ``(n_sample, n_date)`` prediction grid.

        Fills NaN positions with ``0`` (the same dtype as ``pred``).
        """
        pred = self.label.to(self.pred) * 0
        pred[self.finite] = self.pred
        return pred

    def to_dataframe(self):
        df = pd.DataFrame(self.to_2d().cpu().numpy() , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df
    
    def rankic(self):
        assert self.label is not None , f'{self.__class__.__name__} label is None'
        return rankic_2d(self.to_2d() , self.label , 0)
    
    def ic(self):
        assert self.label is not None , f'{self.__class__.__name__} label is None'
        return ic_2d(self.to_2d() , self.label , 0)

@dataclass
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
                        * ``ts_type``           – ``'lin'`` | ``'exp'`` | ``None``
                        * ``cs_type``           – ``'top'`` | ``'ones'`` | ``None``
                        * ``bm_type``           – ``'in'`` | ``None``
                        * ``ts_lin_rate``       – float (default 0.5)
                        * ``ts_half_life_rate`` – float (default 0.5)
                        * ``cs_top_tau``        – float
                        * ``cs_ones_rate``      – float (default 2.0)
                        * ``bm_rate``           – float (default 2.0)
                        * ``bm_secid``          – benchmark security IDs or ``None``
        n_bins:       When not ``None``, ``y`` is converted to integer category
                      labels in ``[0, n_bins - 1]`` for classification training.
    """
    x : torch.Tensor
    y : torch.Tensor | Any = None
    w : torch.Tensor | Any = None
    secid   : np.ndarray | Any = None
    date    : np.ndarray | Any = None
    feature : np.ndarray | Any = None
    weight_param : dict[str,Any] = field(default_factory=dict)
    n_bins : int | None = None
    
    def __post_init__(self):
        self.update_feature()
        self.weight_method = BoostWeightMethod(**self.weight_param)
        self.use_feature = self.feature
        self._raw_y = self.y
        self.to_categorical(n_bins=self.n_bins)

    def __repr__(self):
        return '\n'.join(
            [f'secid={self.secid}', 
             f'date={self.date}', 
             f'feature={self.feature}', 
             f'weight_method={self.weight_method})'])

    @property
    def complete(self):
        return self.y is not None and self.secid is not None and self.date is not None and self.feature is not None
    @property
    def finite(self):
        if self.y is not None:
            return self.y >= 0 if self.is_categorical else self.y.isfinite()
        else:
            return torch.ones_like(self.x[:,:,0] , dtype=torch.bool)
    @property
    def is_categorical(self): return self.n_bins is not None

    def copy(self): return deepcopy(self)

    def to_categorical(self , n_bins : int | None = 100):
        if n_bins is None:
            self.n_bins = None
            self.y = self._raw_y
        elif n_bins is not None and self.n_bins != n_bins:
            rank_pct = self._raw_y.argsort(dim = 0).argsort(dim = 0).where(~self._raw_y.isnan() , torch.nan)
            rank_pct /= rank_pct.nan_to_num().max(dim = 0 , keepdim = True)[0] + 1e-6
            self.y = (rank_pct * n_bins).int().clip(-1 , n_bins-1)
            self.n_bins = n_bins

        return self

    def set_weight_param(self , **weight_param):
        self.weight_method.reset(**weight_param)

    def update_feature(self , use_feature = None):
        if use_feature is None:
            self.use_feature = self.feature
        else:
            assert all(np.isin(use_feature , self.feature)) , np.setdiff1d(use_feature , self.feature)
            self.use_feature = use_feature

    def obj_flatten(self , obj : torch.Tensor | np.ndarray | None , dropna = True , date_first = True) -> Any:
        """Flatten a 2-D or 3-D array to 1-D (or 2-D for ``x``) by the finite mask.

        Parameters
        ----------
        obj:        Tensor/array aligned with ``(n_sample, n_date, ...)``.
        dropna:     If ``True`` (default) keep only positions where
                    ``self.finite`` is ``True``; otherwise keep all.
        date_first: If ``True`` (default) transpose to ``(n_date, n_sample, ...)``
                    before flattening so that the output is date-major.

        Returns
        -------
        Flattened array of length ``n_finite`` (or ``n_sample * n_date`` when
        ``dropna=False``).
        """
        if obj is None:
            return obj

        finite = self.finite if dropna else self.finite.fill_(True)

        if date_first and obj.ndim > 1:
            obj = obj.transpose(1,0) if isinstance(obj , torch.Tensor) else obj.swapaxes(1,0)
            if finite.ndim > 1: 
                finite = finite.transpose(1,0)

        if obj.ndim == 1:
            finite = finite.flatten()
        return obj[finite]

    def SECID(self , dropna = True):
        """Return flat security-ID array aligned with the finite mask."""
        return self.obj_flatten(self.secid[:,None].repeat(len(self.date),axis=1) , dropna=dropna)

    def DATE(self , dropna = True):
        """Return flat date array aligned with the finite mask."""
        return self.obj_flatten(self.date[None,:].repeat(len(self.secid),axis=0) , dropna=dropna)

    def X(self) -> torch.Tensor:
        """Return flat feature matrix of shape ``(n_finite, n_use_feature)``.

        Only the columns in ``use_feature`` are returned; NaN rows are dropped.
        """
        return self.obj_flatten(self.x[...,match_slice(self.use_feature , self.feature)] , dropna=True)

    def Y(self):
        """Return flat label vector of length ``n_finite``, NaN rows dropped."""
        return self.obj_flatten(self.y , dropna=True)

    def W(self):
        """Return flat sample weight vector of length ``n_finite``, NaN rows dropped.

        If ``self.w`` is ``None`` the weights are computed on-the-fly from
        ``weight_method.calculate_weight()``.
        """
        w = self.weight_method.calculate_weight(self.y , self.secid) if self.w is None and self.y is not None else self.w
        return self.obj_flatten(w , dropna=True)
    
    def Dataset(self , *args): 
        return self.BoostDataset(self.X() , self.Y() , self.W() , self.DATE())
    @dataclass(slots=True)
    class BoostDataset:
        x : Any
        y : Any
        w : Any
        date : Any = None
        
        def as_numpy(self):
            [setattr(self , attr , getattr(self , attr).cpu().numpy()) for attr in self.__slots__ 
             if isinstance(getattr(self , attr) , torch.Tensor)]
            return self

        def to(self , device : torch.device | None = None):
            [setattr(self , attr , getattr(self , attr).to(device)) for attr in self.__slots__ 
             if isinstance(getattr(self , attr) , torch.Tensor)]
            return self
        
        def boost_inputs(self , boost_type : Literal['lgbm' , 'xgboost' , 'catboost'] , group = False):
            self.as_numpy()
            boost_inputs = {'data' : self.x , 'label' : self.y}
            if boost_type == 'lgbm':
                # if group: boost_inputs['group'] = self.group_arr()
                boost_inputs['weight'] = self.w
            elif boost_type == 'xgboost':
                if False and group: 
                    g = self.group_arr()
                    boost_inputs['group'] = g
                    boost_inputs['weight'] = np.array([self.w[e-length:e].sum() for e , length in zip(g.cumsum() , g)])
                boost_inputs['weight'] = self.w
            elif boost_type == 'catboost':    
                # if group: boost_inputs['group_id'] = self.group_id()
                boost_inputs['weight'] = self.w
            else:
                raise ValueError(f'Boost type {boost_type} not supported')
            return boost_inputs
        
        def group_arr(self):
            assert (np.diff(self.date) >= 0).all() , 'date must be sorted'
            return np.unique(self.date , return_counts=True)[1]
        
        def group_id(self): return self.date

        @property
        def nfeat(self): return self.x.shape[-1]
        
    def output(self , pred : torch.Tensor | np.ndarray | Any):
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

        finite = self.finite if self.y is not None else torch.full_like(new_pred , fill_value=True)
        return BoostOutput(new_pred , self.secid , self.date , finite , self._raw_y)

    def pred_to_dataframe(self , pred : np.ndarray | torch.Tensor):
        new_pred = pred.numpy() if isinstance(pred , torch.Tensor) else pred 
        df = pd.DataFrame(new_pred , columns = self.date).assign(secid=self.secid).reset_index().\
            melt(id_vars='secid',var_name='date').set_index(['date','secid'])
        return df

    @property
    def feat_idx(self): 
        return match_values(self.use_feature , self.feature)

    @property
    def shape(self): 
        return self.x.shape

    @property
    def nfeat(self): 
        return len(self.feature) if self.use_feature is None else len(self.use_feature)

    def to_dataframe(self):
        df = pd.DataFrame(self.X().cpu().numpy() , columns = self.feature)
        df['secid'] = self.SECID()
        df['date']  = self.DATE()
        df['label'] = self.Y()
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
        data = data.set_index([var_sec[0] , var_date[0]])

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
    def from_numpy(cls , x : np.ndarray , y : np.ndarray | Any = None,  w : np.ndarray | Any = None ,
                   secid : Any = None , date : Any = None , feature : Any = None ,
                   weight_param : dict[str,Any] = {}):
        """Construct from NumPy arrays, delegating to :meth:`from_tensor`."""
        return cls.from_tensor(torch.Tensor(x) , None if y is None else torch.Tensor(y) ,
                               None if w is None else torch.Tensor(w) ,
                               secid , date , feature , weight_param)
    
    @classmethod
    def from_tensor(cls , x : torch.Tensor , y : torch.Tensor | Any = None , w : torch.Tensor | Any = None ,
                    secid : Any = None , date : Any = None , feature : Any = None ,
                    weight_param : dict[str,Any] = {}):
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
    def concat(cls , datas : 'list[BoostInput | None]'):
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
            y[*tar_grid] = blk.y[*src_grid]
            if blk.w is not None and w is not None:
                w[*tar_grid] = blk.w[*src_grid]

        new_binput = cls(x , y , w , secid , date , feature)
        return new_binput
        
    
@dataclass(slots=True)
class BoostWeightMethod:
    """Three-axis multiplicative sample weight calculator.

    Computes ``w = cs_weight * ts_weight * bm_weight`` element-wise over a
    ``(n_sample, n_date)`` grid.

    Attributes:
        ts_type:           Time-series weighting scheme.
                           ``'lin'`` — linearly increasing from ``ts_lin_rate``
                           to 1 across dates.
                           ``'exp'`` — exponential decay with half-life
                           ``ts_half_life_rate * n_date``.
        cs_type:           Cross-sectional weighting scheme.
                           ``'ones'`` — doubles weight on positive-label samples.
                           ``'top'`` — exponential rank-based upweighting.
        bm_type:           Benchmark membership weighting.
                           ``'in'`` — doubles weight for securities in
                           ``bm_secid``.
        ts_lin_rate:       Start value of linear time-series weights (default 0.5).
        ts_half_life_rate: Half-life as a fraction of ``n_date`` (default 0.5).
        cs_top_tau:        Decay exponent for the ``'top'`` cross-sectional scheme.
        cs_ones_rate:      Multiplier applied to positive-label rows (default 2.0).
        bm_rate:           Multiplier applied to benchmark members (default 2.0).
        bm_secid:          Security IDs that constitute the benchmark universe.
    """
    ts_type : Literal['lin' , 'exp'] | None = None
    cs_type : Literal['top' , 'positive' , 'ones'] | None = None
    bm_type : Literal['in'] | None = None
    ts_lin_rate : float = 0.5
    ts_half_life_rate : float = 0.5
    cs_top_tau : float = 0.75*np.log(0.5)/np.log(0.75)
    cs_ones_rate : float = 2.
    bm_rate : float = 2.
    bm_secid : np.ndarray | list | None = None

    def calculate_weight(self , y : np.ndarray | torch.Tensor , secid : Any):
        """Compute the combined ``(n_sample, n_date)`` weight matrix.

        The result is the element-wise product of :meth:`cs_weight`,
        :meth:`ts_weight`, and :meth:`bm_weight`.
        """
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y[...,0]
        assert y.ndim == 2 , y.shape
        return self.cs_weight(y) * self.ts_weight(y) * self.bm_weight(y , secid)

    def cs_weight(self , y : np.ndarray | torch.Tensor , **kwargs):
        """Cross-sectional weights of shape ``(n_sample, n_date)``.

        ``'ones'``: samples with label ``== 1`` get weight ``cs_ones_rate``
        (default ×2).
        ``'top'``: exponential rank-based decay so top-ranked securities receive
        higher weight.  ``None``: uniform ones.
        """
        w = y * 0 + 1.
        if self.cs_type is None: 
            return w
        elif self.cs_type == 'ones':
            w[y == 1.] = w[y == 1.] * 2
        elif self.cs_type == 'top':
            for j in range(w.shape[1]):
                v = y[:,j].argsort() + y[:,j] * 0
                w[:,j] = np.exp((1 - v / np.nanmax(v).astype(float))*np.log(0.5) / self.cs_top_tau)
        else:
            raise KeyError(self.cs_type)
        return w
    
    def ts_weight(self , y : np.ndarray | torch.Tensor , **kwargs):
        """Time-series weights of shape ``(n_sample, n_date)``.

        ``'lin'``: linearly increases from ``ts_lin_rate`` to 1 across dates.
        ``'exp'``: exponential decay so recent dates have higher weight.
        ``None``: uniform ones.
        """
        w = y * 0 + 1.
        if self.ts_type is None: 
            return w
        elif self.ts_type == 'lin':
            w *= np.linspace(self.ts_lin_rate,1,w.shape[1]).reshape(1,-1)
        elif self.ts_type == 'exp':
            w *= np.power(2 , -np.arange(w.shape[1])[::-1] / int(self.ts_half_life_rate * w.shape[1])).reshape(1,-1)
        else:
            raise KeyError(self.ts_type)
        return w
    
    def bm_weight(self , y : np.ndarray | torch.Tensor , secid : np.ndarray | list):
        """Benchmark-membership weights of shape ``(n_sample, n_date)``.

        ``'in'``: securities in ``bm_secid`` receive weight ``bm_rate + 1``
        (default ×2), others weight 1.  ``None``: uniform ones.
        """
        w = y * 0 + 1.
        if self.bm_type is None: 
            return w
        elif self.bm_type == 'in': 
            if self.bm_secid is not None:
                w *= np.isin(secid , self.bm_secid) * 1 + 1
        else:
            raise KeyError(self.bm_type)
        return w
    
    def reset(self , **kwargs):
        [setattr(self , k , v) for k,v in kwargs.items()]
