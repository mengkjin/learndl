"""Minchars preprocessors."""
from __future__ import annotations
import os
import numpy as np
import polars as pl
from src.proj import DB, CALENDAR, Base
from src.data.util import DataBlock
from ..core import MicellaneousPreProcessor
from ..mem_trace import log_mem, mesh4d_bytes

class _MinCharsPreProcessor(MicellaneousPreProcessor):
    """
    Load all selected ``min_chars`` columns (daily + roll + tag) into one block.

    Subclasses choose the transform: cross-sectional z-score only, or per-secid
    rolling pct_rank then cross-sectional z-score.  Non-finite values → 0.

    ``DateChunkYears=1`` so fit reconstruct processes one calendar year at a time.
    """
    FEATURE_CHUNK_SIZE = 20
    fit_start = 20100101
    hist_start = 20100101
    DB_SRC = 'min_chars'
    MIN_SAMPLES = 90
    RANK_THEN_CS = False
    DateChunkYears = 1
    ChunkFillNan = 0.0
    MemTrace = os.environ.get('LEARNDL_MEMORY_TRACE') == '1'
    MaxAllocationBytes = 32 * 1024**3

    def pre_process(
        self , start : int | None = None , end : int | None = None , * ,
        secid : Base.alias.SecidType = None ,
        indent = 0 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs
    ) -> DataBlock:
        start = start or self.load_start
        end = end or self.load_end
        if start > end:
            return DataBlock()
        from src.data.update.custom.min_chars._catalog import selected_by_db_key
        grouped = selected_by_db_key()
        if not grouped:
            raise ValueError(f'{self.key}: no selected min_chars columns')
        load_start = CALENDAR.td(start , -self.CalculationWindow + 1).td
        secid = Base.ensure_secid(secid)
        blocks : list[DataBlock] = []
        for db_key , features in grouped.items():
            self.logger.stdout(
                f'{self.key} load {self.DB_SRC}/{db_key} ({len(features)} cols) {load_start}-{end}' ,
                vb = 2 , add_prefix = False ,
            )
            if self.MemTrace:
                log_mem(self.logger, f'{self.key} {db_key} before-read', retained_tables=blocks)
            from src.data.util.minchars_stock import historical_stock_ids
            from ..minchars_input import load_selected
            known = historical_stock_ids()
            paths = DB.paths(self.DB_SRC, db_key, start=load_start, end=end)
            frames = []
            for path in paths:
                daily = load_selected(path, features, known, secid)
                if not daily.is_empty():
                    frames.append(daily)
            df = pl.concat(frames, how='vertical_relaxed') if frames else pl.DataFrame()
            del frames
            if df.is_empty():
                continue
            if self.MemTrace:
                log_mem(self.logger, f'{self.key} {db_key} selected-columns-read', df=df)
            self.logger.stdout(f'{self.key} {db_key} universe={df["secid"].n_unique()} known-historical={len(known)} rows={df.height}',
                               vb=2, add_prefix=False)
            if secid is not None:
                df = df.filter(pl.col('secid').is_in(secid))
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise ValueError(f'{self.DB_SRC}/{db_key} missing selected columns: {missing}')
            df = df.select(['secid' , 'date'] + features)
            if self.MemTrace:
                log_mem(self.logger , f'{self.key} {db_key} loaded' , df = df)
            blk = self._transform_chunks(df , features)
            del df
            if self.MemTrace:
                log_mem(self.logger , f'{self.key} {db_key} transformed' , blk = blk)
            if not blk.empty:
                blocks.append(blk)
            del blk
        if not blocks:
            return DataBlock()
        if self.MemTrace:
            n = len(np.unique(np.concatenate([b.secid for b in blocks])))
            t = len(np.unique(np.concatenate([b.date for b in blocks])))
            i = max(int(b.shape[2]) for b in blocks)
            f = sum(len(b.feature) for b in blocks)
            log_mem(
                self.logger , f'{self.key} tables-merge' ,
                tables = blocks ,
                union = n * t * i * f * 4 ,
                mesh4d = mesh4d_bytes(n , t , i , f) ,
            )
        from ..memory_guard import check_allocation
        union_n = len(np.unique(np.concatenate([b.secid for b in blocks])))
        union_t = len(np.unique(np.concatenate([b.date for b in blocks])))
        union_f = len(np.unique(np.concatenate([b.feature for b in blocks])))
        check_allocation(union_n * union_t * union_f * 4 + max(b.values.numel() * b.values.element_size() for b in blocks),
                         f'{self.key} tables merge', limit=self.MaxAllocationBytes)
        block = DataBlock.merge(blocks , inplace = True).slice_date(start , end).fillna(0)
        del blocks
        if self.MemTrace:
            log_mem(self.logger , f'{self.key} year-block' , block = block)
        if not self.DateChunkYears:
            block = block.mask_values(mask = self.mask)
        return block

    def _transform_chunks(self , df : pl.DataFrame , features : list[str]) -> DataBlock:
        """Z-score (and optional rolling rank) in feature chunks; concat along feature."""
        acc = DataBlock()
        for i in range(0 , len(features) , self.FEATURE_CHUNK_SIZE):
            sub = features[i : i + self.FEATURE_CHUNK_SIZE]
            lf = df.lazy().select(['secid' , 'date'] + sub)
            if self.RANK_THEN_CS:
                lf = self._rolling_pct_rank(lf , sub)
            lf = self._cs_zscore_fill0(lf , sub)
            chunk = DataBlock.from_polars(lf.collect())
            if chunk.empty:
                continue
            acc = chunk if acc.empty else DataBlock.concat_feature([acc , chunk])
            del chunk
            if self.MemTrace:
                log_mem(self.logger , f'{self.key} feat-chunk {sub[0]}..{sub[-1]}' , acc = acc)
        return acc

    def _rolling_pct_rank(self , lf : pl.LazyFrame , features : list[str]) -> pl.LazyFrame:
        """Per-secid rolling rank / window length, same definition as ``PrePro_dfl2``."""
        rank_names = [f'{feat}_rk' for feat in features]
        return (
            lf.sort(['secid' , 'date'])
            .with_columns(pl.lit(1).alias('_one'))
            .with_columns(
                pl.col(feat).rolling_rank(
                    window_size = self.CalculationWindow , min_samples = self.MIN_SAMPLES ,
                ).over('secid').alias(f'{feat}_rk')
                for feat in features
            )
            .with_columns(
                pl.col('_one').rolling_sum(window_size = self.CalculationWindow).over('secid').alias('_wn')
            )
            .with_columns((pl.col(f'{feat}_rk') / pl.col('_wn')).alias(feat) for feat in features)
            .drop(['_one' , '_wn'] + rank_names)
        )

    @staticmethod
    def _cs_zscore_fill0(lf : pl.LazyFrame , features : list[str]) -> pl.LazyFrame:
        """Per-date z-score; non-finite (NaN/inf, empty cross-section) → 0."""
        exprs = []
        for feat in features:
            z = (
                (pl.col(feat) - pl.col(feat).mean().over('date'))
                / (pl.col(feat).std().over('date') + 1e-6)
            )
            exprs.append(pl.when(z.is_finite()).then(z).otherwise(0.0).cast(pl.Float32).alias(feat))
        return lf.with_columns(exprs)

class PrePro_minc(_MinCharsPreProcessor):
    """
    All selected ``min_chars`` columns with cross-sectional z-score (key: ``'minc'``).

    Same transform as ``dfl2cs``: per-date mean/std, then missing/non-finite → 0.
    """

class PrePro_mincr(_MinCharsPreProcessor):
    """
    All selected ``min_chars`` columns with rolling pct_rank then CS z-score (key: ``'mincr'``).

    Per-secid rolling percentage rank (window ``CalculationWindow``, min ``MIN_SAMPLES``),
    then the same cross-sectional z-score and fill-0 as ``PrePro_minc``.
    """
    CalculationWindow = 250
    RANK_THEN_CS = True

