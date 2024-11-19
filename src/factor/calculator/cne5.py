import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Any , Literal , Optional

from src.basic import PATH , CONF
from src.data import DATAVENDOR
from src.func.transform import (time_weight , descriptor , apply_ols , neutral_resid , ewma_cov , ewma_sd)

def parse_ts_input(
    ts : pd.DataFrame , 
    value_cols = ['value'] , 
    date_cols = ['date' , 'end_date'] , 
    feat_cols = ['secid' , 'factor']
):
    ts_cols = np.array([*ts.index.names , *ts.columns]).astype(str)
    value_cols = np.intersect1d(value_cols , ts_cols)
    date_cols  = np.intersect1d(date_cols  , ts_cols)
    feat_cols  = np.intersect1d(feat_cols  , ts_cols)
    assert len(value_cols) == 1 and len(date_cols) == 1 and len(feat_cols) == 1 , (value_cols , date_cols , feat_cols)
    ts = ts.pivot_table(value_cols[0] , date_cols[0] , feat_cols[0])
    return ts.to_numpy() , ts.columns.to_numpy()
    
def parse_cov_output(cov : np.ndarray , feat : np.ndarray):
    return pd.DataFrame(cov , index=feat , columns=feat)

def parse_sd_output(sd : np.ndarray , feat : np.ndarray , feat_name = 'sd'):
    return pd.DataFrame(sd , index=feat , columns=[feat_name])

class DateDfs:
    def __init__(self , max_len = 50) -> None:
        self.max_len = max_len
        self.D : dict[int , pd.DataFrame] = {}

    def trim(self , date : Optional[int] = None):
        if date and date in self.D: del self.D[date] 
        if len(self.D) >= self.max_len: del self.D[list(self.D.keys())[0]]

    def add(self , v , date : int):
        self.trim(date)
        self.D[date] = v

    def get(self , date : int):
        if date in self.D: return self.D[date]
        else: return None

class DateSeriesDict:
    def __init__(self , max_len = 50) -> None:
        self.max_len = max_len
        self.D : dict[int , dict[str , pd.Series]] = {}

    def trim(self , date : Optional[int] = None):
        if len(self.D) >= self.max_len: del self.D[list(self.D.keys())[0]]
        if date not in self.D and date is not None: self.D[date] = {}

    def add(self , v , date : int ,  name : str):
        self.trim(date)
        self.D[date][name] = v

    def get(self , date : int , name : str):
        if date in self.D and name in self.D[date]: 
            return self.D[date][name]
        else: return None   

class TuShareCNE5_Calculator:
    START_DATE = 20050101
    def __init__(self) -> None:

        self.estuniv = DateDfs(50)
        self.ind_grp = DateDfs(50)
        self.ind_exp = DateDfs(50)
        self.style   = DateSeriesDict(50)
        self.exposure = DateDfs(50)

        self.coef = DateDfs(756)
        self.resid = DateDfs(756)

        self.common_risk   = DateDfs(50)
        self.specific_risk = DateDfs(50)

    def descriptor(self , v , date : int , name : str , fillna : Any = 0) -> pd.Series:
        assert isinstance(v , pd.Series) , v
        univ = self.get_estuniv(date)
        indus = self.ind_grp.get(date)
        if indus is None or indus.empty:
            self.calc_indus(date)
            indus = self.ind_grp.get(date)
        v = descriptor(v.reindex(univ.index) , univ['weight'] , fillna , indus)
        v.name = name
        self.style.add(v , date , name)
        return v

    def get_exposure(self , date : int , read = False):
        df = self.exposure.get(date)
        if (df is None or df.empty) and read: 
            df = PATH.db_load('models' , 'tushare_cne5_exp' , date)
            if 'secid' in df.columns: df = df.set_index('secid')
        if df is None or df.empty: 
            df = pd.concat([self.get_estuniv(date).loc[:,['estuniv','weight','market']] , 
                            self.get_industry(date) , 
                            *[self.get_style(date , name) for name in CONF.RISK_STYLE]] , axis=1)
        df = df.loc[:,['estuniv' , 'weight'] + CONF.RISK_COMMON]
        self.exposure.add(df , date)
        return df

    def get_estuniv(self , date : int):
        df = self.estuniv.get(date)
        if df is None or df.empty: df = self.calc_estuniv(date)
        return df
    
    def get_industry(self , date : int):
        df = self.ind_exp.get(date)
        if df is None or df.empty: df = self.calc_indus(date)
        return df
    
    def get_style(self , date : int , name : str):
        assert name in CONF.RISK_STYLE , name
        df = self.style.get(date , name)
        if df is None or df.empty: df = getattr(self , f'calc_{name}')(date)
        return df
    
    def get_coef(self , date : int , read = False):
        coef = self.coef.get(date)
        if (coef is None or coef.empty) and read: 
            coef = PATH.db_load('models' , 'tushare_cne5_coef' , date)
            self.coef.add(coef , date)
        if coef is None or coef.empty: 
            coef , resid = self.calc_model(date)
        return coef
    
    def get_resid(self , date : int , read = False):
        resid = self.resid.get(date)
        if (resid is None or resid.empty) and read: 
            resid = PATH.db_load('models' , 'tushare_cne5_res' , date)
            self.resid.add(resid , date)
        if resid is None or resid.empty: 
            coef , resid = self.calc_model(date)
        return resid

    def calc_estuniv(self , date : int):
        list_days = 252
        redempt_tmv_pct = 0.8

        dates = DATAVENDOR.CALENDAR.td_trailing(date , 63)

        new_desc = DATAVENDOR.INFO.get_desc(date)
        abnormal = DATAVENDOR.INFO.get_abnormal(date)
        new_list_dt = DATAVENDOR.INFO.get_list_dt(date , list_days)

        trd = DATAVENDOR.TRADE.get_trd(DATAVENDOR.CALENDAR.td(date , -21)).loc[:,['secid','status']]
        trd = DATAVENDOR.TRADE.get_trd(date).loc[:,['secid','status']].merge(trd , on = 'secid' , how = 'left').\
            set_index('secid').reindex(new_desc.index).fillna(0)
        
        val = pd.concat([DATAVENDOR.TRADE.get_val(d , ['secid','date','circ_mv','total_mv']) for d in dates]).\
            sort_values(['secid','date']).set_index('secid').groupby('secid').ffill().\
            groupby('secid').last().reindex(new_desc.index)
        
        # trading status are 1.0 this day or 1 month ealier
        rule0 = pd.concat([DATAVENDOR.TRADE.get_trd(d , ['secid','date','status']) for d in dates]).\
            groupby('secid')['status'].sum().reindex(new_desc.index) > 0

        # list date 1 year earlier and not delisted or total mv in the top 20%
        rule1 = ((new_desc['delist_dt'] > date) & (new_list_dt['list_dt'] <= date)) | \
            (val['total_mv'].rank(pct = True , na_option='bottom') >= redempt_tmv_pct)
        # not st
        rule2 = ~new_desc.index.isin(abnormal['secid'])

        # total_mv not nan
        rule3 = val['total_mv'] > 0

        new_desc['estuniv'] = 1 * (rule0 & rule1 & rule2 & rule3)
        new_desc['weight'] = val['circ_mv'].fillna(0) / 1e8
        new_desc['market'] = 1.0
        self.estuniv.add(new_desc , date)
        return new_desc
    
    def calc_indus(self , date : int):
        univ = self.get_estuniv(date)
        df = DATAVENDOR.INFO.get_indus(date)
        self.ind_grp.add(df , date)

        df = df.assign(values = 1).pivot_table('values' , 'secid' , 'indus', fill_value=0).loc[:,CONF.RISK_INDUS]
        df = df.reindex(univ.index).fillna(0).rename_axis(columns=None)
        self.ind_exp.add(df , date)
        
        return df
    
    def calc_style(self , date : int):
        for style_name in CONF.RISK_STYLE: getattr(self , f'calc_{style_name}')(date)
    
    def calc_size(self , date : int):
        v = np.log(DATAVENDOR.TRADE.get_val(date).set_index('secid')['total_mv'] / 10**8)
        return self.descriptor(v , date , 'size' , 'min')
    
    def calc_beta(self , date : int):
        n_window = 252
        half_life = 63
        min_finite_ratio = 0.25

        dates = DATAVENDOR.CALENDAR.td_trailing(date , n_window)
        
        stk_ret = DATAVENDOR.TRADE.get_returns(dates[0], dates[-1] , mask = False)
        mkt_ret = DATAVENDOR.TRADE.get_market_return(dates[0], dates[-1])
        
        wgt = time_weight(n_window , half_life)

        b = apply_ols(mkt_ret.values.flatten() , stk_ret.values , wgt)[1]
        b[np.isfinite(stk_ret).sum() < n_window * min_finite_ratio] = np.nan
        v = pd.Series(b , index = stk_ret.columns)

        return self.descriptor(v , date , 'beta' , 0)

    def calc_momentum(self , date : int):
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 525)[:504]
        wgt_df = pd.DataFrame({'date':dates , 'weight':time_weight(504,126)})

        df = pd.concat([DATAVENDOR.TRADE.get_trd(d , ['date','secid','pctchange']) for d in dates]).merge(wgt_df , on = 'date')
        df['lnret'] = np.log(1 + df['pctchange'] / 100) * df['weight']
        v = df.groupby('secid')['lnret'].sum()
        
        return self.descriptor(v , date , 'momentum' , 0)
    
    def calc_residual_volatility(self , date : int):
        # 0.74 * dastd + 0.16 * cmra + 0.10 * hsigma
        # orthogonalize over size and beta
        # dsastd : annualized daily standard deviation of 252 trade_days with half_life 42
        # cmra : cumulative log range , (zmax - zmin) over the last 12 months
        # hsigma : annualized daily standard deviation of daily residual return (same parameters as beta calculation)

        dates = DATAVENDOR.CALENDAR.td_trailing(date , 253)[:-1]
        wgt = time_weight(252 , 42)

        df_trd = pd.concat([DATAVENDOR.TRADE.get_trd(d,['date','secid','pctchange']) for d in dates])
        df_trd = df_trd.pivot_table('pctchange','date','secid') / 100
        dsastd = self.descriptor((df_trd * wgt.reshape(-1,1)).std() , date , 'dsastd' , 'median')

        df_cum = (df_trd.fillna(0) + 1).cumprod()
        cmra = self.descriptor(df_cum.max() - df_cum.min() , date , 'cmra' , 'median')

        wgt = time_weight(252 , 63)

        res_list = [DATAVENDOR.RISK.get_res(d) for d in dates]
        if len([res for res in res_list if res is not None]) >= 63:
            df_res = pd.concat([DATAVENDOR.RISK.get_res(d) for d in dates])
            df_res = df_res.pivot_table('resid','date','secid').reindex(dates)
            hsigma = self.descriptor((df_res * wgt.reshape(-1,1)).std() , date , 'hsigma' , 'median')
        else:
            hsigma = 0

        resvol = (0.74 * dsastd +  0.16 * cmra + 0.10 * hsigma).rename('resvol')

        x = pd.concat([self.get_style(date , 'size') , self.get_style(date , 'beta')] , axis=1)
        model = np.linalg.lstsq(x , resvol.reindex(x.index).fillna(resvol.median()) , rcond=None)
        v = (resvol - x @ model[0])

        return self.descriptor(v , date , 'residual_volatility' , 'median')
    
    def calc_non_linear_size(self , date : int):
        size = self.get_style(date , 'size')
        v = neutral_resid(size , size ** 3 , np.sqrt(self.get_estuniv(date)['weight']))
        return self.descriptor(v , date , 'non_linear_size' , 'min')
    
    def calc_book_to_price(self , date : int):
        v = (1 / DATAVENDOR.TRADE.get_val(date)['pb']).fillna(0)
        return self.descriptor(v , date , 'book_to_price' , 'median')
    
    def calc_liquidity(self , date : int):

        cols = ['secid','turnover_rate']
        stom = pd.concat([DATAVENDOR.TRADE.get_val(d , cols) for d in DATAVENDOR.CALENDAR.td_trailing(date , 21)]).\
            groupby('secid')['turnover_rate'].sum()
        stom = self.descriptor(stom.fillna(0) , date , 'stom' , 'min')
        
        stoq = pd.concat([DATAVENDOR.TRADE.get_val(d , cols) for d in DATAVENDOR.CALENDAR.td_trailing(date , 63)]).\
            groupby('secid')['turnover_rate'].sum()
        stoq = self.descriptor(stoq.fillna(0) , date , 'stoq' , 'min')
        
        stoa = pd.concat([DATAVENDOR.TRADE.get_val(d , cols) for d in DATAVENDOR.CALENDAR.td_trailing(date ,252)]).\
            groupby('secid')['turnover_rate'].sum()
        stoa = self.descriptor(stoa.fillna(0) , date , 'stoa' , 'min')

        v = 0.35 * stom + 0.35 * stoq + 0.3 * stoa

        return self.descriptor(v , date , 'liquidity' , 'median')
    
    def calc_earnings_yield(self , date : int):
        cp = DATAVENDOR.TRADE.get_trd(date , ['secid' , 'close']).set_index('secid')
        cetop = DATAVENDOR.INDI.get_ttm('ocfps' , date , 1)['ocfps'] / cp['close']
        cetop = self.descriptor(cetop.fillna(0) , date , 'cetop' , 'median')

        etop  = 1 / DATAVENDOR.TRADE.get_val(date , ['secid' , 'pe']).set_index('secid')['pe']
        etop = self.descriptor(etop.fillna(0) , date , 'etop' , 'median')

        v = 0.21 * cetop + 0.79 * etop

        return self.descriptor(v , date , 'earnings_yield' , 'median')
    
    def calc_growth(self , date : int):

        val = 'diluted2_eps'
        df = DATAVENDOR.INDI.get_acc(val , date , 6 , year_only=True).groupby('secid').tail(5).copy()
        df = df.assign(idx = df.groupby('secid').cumcount()).pivot_table(val , 'idx' , 'secid')
        df = pd.DataFrame({'secid':df.columns,'value':apply_ols(df.index.values,df.values)[1],'na':np.isnan(df.values).sum(axis=0)})
        egro = df[df['na'] <= 1].set_index('secid')['value']
        egro = self.descriptor(egro.fillna(0) , date , 'egro' , 'median')

        val = 'revenue_ps'
        df = DATAVENDOR.INDI.get_acc(val , date , 6 , year_only=True).groupby('secid').tail(5).copy()
        df = df.assign(idx = df.groupby('secid').cumcount()).pivot_table(val , 'idx' , 'secid')
        df = pd.DataFrame({'secid':df.columns,'value':apply_ols(df.index.values,df.values)[1],'na':np.isnan(df.values).sum(axis=0)})
        sgro = df[df['na'] <= 1].set_index('secid')['value']
        sgro = self.descriptor(sgro.fillna(0) , date , 'sgro' , 'median')

        v = 0.53 * egro + 0.47 * sgro
        return self.descriptor(v , date , 'growth' , 'median')
    
    def calc_leverage(self , date : int):

        cp = DATAVENDOR.TRADE.get_trd(date , ['secid' , 'close']).set_index('secid')
        mlev = (DATAVENDOR.INDI.get_acc('longdeb_to_debt' , date , 1)['longdeb_to_debt'].fillna(100) / 100 *
            DATAVENDOR.INDI.get_acc('debt_to_eqt' , date , 1)['debt_to_eqt'] / 100 *
            DATAVENDOR.INDI.get_acc('bps' , date , 1)['bps'] / cp['close'])
        mlev = self.descriptor(mlev , date , 'mlev' , 'median')

        dtoa = DATAVENDOR.INDI.get_acc('debt_to_assets' , date , 1)['debt_to_assets']
        dtoa = self.descriptor(dtoa , date , 'dtoa' , 'median')

        blev = DATAVENDOR.INDI.get_acc('assets_to_eqt' , date , 1)['assets_to_eqt']
        blev = self.descriptor(blev , date , 'blev' , 'median')

        v = 0.38 * mlev + 0.35 * dtoa + 0.27 * blev

        return self.descriptor(v , date , 'leverage' , 'median')
    
    def calc_model(self , date : int):
        exp_date = DATAVENDOR.CALENDAR.td(date , -1).as_int() # DATAVENDOR.CALENDAR.offset(date , -1)
        exp = self.get_exposure(exp_date , read = True)
        exp = exp[exp['estuniv'] == 1]
        ret = DATAVENDOR.TRADE.get_trd(date , ['secid','pctchange']).set_index('secid') / 100
        ret = ret.reindex(exp.index).fillna(0).rename(columns={'pctchange':'ret'})

        wgt : Any = exp['weight']
        mkt = (exp['estuniv'] * exp['market']).rename('market')
        rsk = exp.drop(columns=['estuniv','weight','market'])
        mkt_model = sm.WLS(ret[['ret']] , mkt , weights = wgt).fit()
        rsk_model = sm.WLS(mkt_model.resid , rsk , weights = wgt).fit()

        coef = pd.concat([
            pd.concat([mkt_model.params , rsk_model.params]) ,
            pd.concat([mkt_model.tvalues , rsk_model.tvalues])] , 
            axis = 1).rename(columns={0:'coef',1:'tvalue'})
        resid = rsk_model.resid.rename('resid').to_frame()

        self.coef.add(coef , date)
        self.resid.add(resid , date)
        return coef , resid
    
    def calc_common_risk(self , date : int):
        assert date >= self.START_DATE , (date , self.START_DATE)
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 504)
        dates = dates[dates >= self.START_DATE]
        if len(dates) < (504 // 4): 
            factors = self.get_coef(date,True).index.to_numpy()
            cov = pd.DataFrame(None , index=factors , columns = factors).reset_index().rename(columns={'index':'factor_name'})
            return cov

        coefs = pd.concat([self.get_coef(d,True).assign(date = d) for d in dates]).reset_index().\
            rename(columns={'index':'factor','coef':'value'})
        ts , feat = parse_ts_input(coefs)
        corr = ewma_cov(ts , 504 , 180 , 0.33 , True)
        sd   = ewma_sd(ts , 504 , 90)
        cov  = parse_cov_output(sd[:,None].dot(sd[None]) * corr , feat)
        cov  = cov.reset_index().rename(columns={'index':'factor_name'})
        self.common_risk.add(cov , date)
        return cov

    def calc_specific_risk(self , date : int):
        assert date >= self.START_DATE , (date , self.START_DATE)
        dates = DATAVENDOR.CALENDAR.td_trailing(date , 504)
        dates = dates[dates >= self.START_DATE]
        if len(dates) < (504 // 4): 
            secids = self.get_resid(date,True).index.to_numpy()
            sd = pd.DataFrame(None , index=secids , columns=['spec_risk']).reset_index().rename(columns={'index':'secid'})
            return sd

        resids = pd.concat([self.get_resid(d,True).assign(date = d) for d in dates]).reset_index().\
            rename(columns={'resid':'value'})
        ts , feat = parse_ts_input(resids)
        sd = parse_sd_output(ewma_sd(ts , 504 , 90) , feat , 'spec_risk')
        sd = sd.reset_index().rename(columns={'index':'secid'})
        self.specific_risk.add(sd , date)
        return sd
    
    def Update(self , job : Literal['exposure' , 'risk'] , start : int | Any = None , end : int | Any = None):
        dates = self.updatable_dates(job)
        if start: dates = dates[dates >= start]
        if end  : dates = dates[dates <= end]
        for date in dates: 
            self.update_date(date , job)
        return self
        
    def updatable_dates(self , job : Literal['exposure' , 'risk']):
        dates = DATAVENDOR.CALENDAR.trade_dates()
        end_date = np.min([PATH.db_dates('trade_ts' , 'day').max(), PATH.db_dates('trade_ts' , 'day_val').max()])
        dates = dates[(dates > self.START_DATE) & (dates <= end_date)]
        
        all_updated : np.ndarray | Any = None
        if job == 'exposure':
            check_list = ['tushare_cne5_exp','tushare_cne5_coef','tushare_cne5_res']
        elif job == 'risk':
            check_list = ['tushare_cne5_cov','tushare_cne5_spec']
        else:
            raise KeyError(job)
        
        for x in check_list:
            updated = PATH.db_dates('models' , x)
            all_updated = updated if all_updated is None else np.intersect1d(all_updated , updated)
        return np.setdiff1d(dates , all_updated)
        
    def update_date(self , date : int , job : Literal['exposure' , 'risk']):
        assert DATAVENDOR.CALENDAR.is_trade_date(date) , f'{date} is not a trade_date'
        if job == 'exposure':
            PATH.db_save(self.get_exposure(date) , 'models' , 'tushare_cne5_exp'  , date , verbose=True)
            PATH.db_save(self.get_coef(date)     , 'models' , 'tushare_cne5_coef' , date , verbose=True)
            PATH.db_save(self.get_resid(date)    , 'models' , 'tushare_cne5_res'  , date , verbose=True)
        elif job == 'risk':
            PATH.db_save(self.calc_common_risk(date)   , 'models' , 'tushare_cne5_cov'  , date , verbose=True)
            PATH.db_save(self.calc_specific_risk(date) , 'models' , 'tushare_cne5_spec' , date , verbose=True)
        else:
            raise KeyError(job)

#date = 20120105
#tushare_cne5 = TuShareCNE5_Calculator()
#tushare_cne5.Update(20110101 , 20151231 , ['exposure'])
