import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py , time
from learndl.scripts.data_utils.gen_data import save_data_file
from ..util.environ import get_logger
from datetime import datetime,timedelta
from ..functional.func import *

logger = get_logger()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dir_data      = f'../../data/'
dir_fund_port = f'{dir_data}/fund_stock_port.h5'
dir_trading   = f'{dir_data}/day_trading_data.h5'
dir_info      = f'{dir_data}/stock_info.h5'

class matrix_factorization():
    def __init__(self , m , learn_rates = [0.1,0.05,0.01,0.005,0.001]):
        self.mat = m.to(DEVICE)
        self.nrow = m.shape[0]
        self.ncol = m.shape[1]
        self.learn_rates = learn_rates
    
    def proceed(self , nfeat , ltheta = 0.01 , num_epochs = 200 , print_process = False):
        self.mod = []
        self.losses = []
        for learn_rate in self.learn_rates:
            mod = self.factor_module((self.nrow , self.ncol , nfeat)).to(DEVICE)
            criterion = nn.MSELoss(reduction = 'sum')
            optimizer = torch.optim.Adam(mod.parameters(), lr = learn_rate)

            losses = []  
            for epoch in range(num_epochs):
                output , theta = mod()
                loss = criterion(output, self.mat) + ltheta * theta

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                # if (epoch) % 10 == 0: print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, loss.item()))
            
            if print_process : print('Learn rate {:.5f}, Epoch [{}/{}], Loss: {:.8f}'.format(learn_rate, epoch+1, num_epochs, loss.item()))
            self.mod.append(mod)
            self.losses.append(losses)
        
        best_mod = np.argmin([l[-1] for l in self.losses])
        if print_process:
            plt.plot(range(num_epochs), self.losses[best_mod])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training Loss , Learn Rate: {self.learn_rates[best_mod]}')
            plt.show()

        self.Q = torch.cat((self.mod[best_mod].Q_bias.reshape(-1,1) , self.mod[best_mod].Q_weight) , dim = 1).detach().cpu().numpy()
        self.m_pred = output
        self.theta = theta
    
    class factor_module(nn.Module):
        def __init__(self , ndim):
            super().__init__()
            self.nf , self.ns , self.nd = ndim
            """
            P_weight = torch.rand(self.nf , self.nd - 1) / np.sqrt(self.nf)
            P_bias   = torch.zeros(self.nf)
            Q_weight = torch.rand(self.ns , self.nd - 1) / np.sqrt(self.ns)
            Q_bias   = torch.zeros(self.ns)
            bias0    = torch.zeros(1)

            self.P_weight = torch.nn.Parameter(P_weight)
            self.P_bias   = torch.nn.Parameter(P_bias)
            self.Q_weight = torch.nn.Parameter(Q_weight)
            self.Q_bias   = torch.nn.Parameter(Q_bias)
            self.bias0    = torch.nn.Parameter(bias0)
            """
            self.P_weight = torch.nn.Parameter(torch.rand(self.nf , self.nd - 1) / np.sqrt(self.nf))
            self.P_bias   = torch.nn.Parameter(torch.zeros(self.nf))
            self.Q_weight = torch.nn.Parameter(torch.rand(self.ns , self.nd - 1) / np.sqrt(self.ns))
            self.Q_bias   = torch.nn.Parameter(torch.zeros(self.ns))
            self.bias0    = torch.nn.Parameter(torch.zeros(1))

        def forward(self):
            # (bat_size, seq, features)
            output = torch.matmul(self.P_weight , self.Q_weight.T) + self.Q_bias + self.P_bias.repeat_interleave(self.ns).reshape(-1,self.ns) + self.bias0
            theta  = sum([p.square().sum() for p in [self.P_weight , self.P_bias , self.Q_weight , self.Q_bias]])
            return output , theta
        
class fund_stock():
    def __init__(self , FundID = None , SecID = None , nfeat = 32 , print_process = False):
        self.port_file = h5py.File(dir_fund_port , mode='r')
        self.FactorDate = np.array(list(self.port_file.keys())).astype(int)
        self.FundID = FundID
        self.SecID = SecID
        self.nfeat = nfeat
        self.print_process = print_process
        
    def calculate_factors(self , date):
        tb = pd.DataFrame(self.port_file[str(date)][:])
        tb.columns = ['FundID','SecID','weight']
        tb.FundID = [self._IDconvert(s) for s in tb.FundID]
        tb.SecID = [self._IDconvert(s) for s in tb.SecID]
        
        target_FundID = sorted(tb.FundID[tb.FundID > 0].unique()) if self.FundID is None else np.array(self.FundID)
        target_SecID = sorted(tb.SecID[tb.SecID > 0].unique()) if self.SecID is None else np.array(self.SecID)
        
        tb = tb[tb.FundID.isin(target_FundID) & tb.SecID.isin(target_SecID)]
        tb = pd.concat([pd.DataFrame({'FundID':target_FundID,'SecID':target_SecID[0],'weight':0.,}) ,
                        pd.DataFrame({'FundID':target_FundID[0],'SecID':target_SecID,'weight':0.,}) , 
                        tb,
                       ])
        self.wide_table = tb.pivot_table('weight','FundID','SecID',aggfunc=sum,fill_value=0.).loc[target_FundID,target_SecID]
        assert np.array_equal(self.wide_table.columns.tolist() , target_SecID)
        assert np.array_equal(self.wide_table.index.tolist() , target_FundID)
        self.factorize(self.nfeat , self.print_process)
        
    def factorize(self , nfeat = 32 , print_process = False):
        mf = matrix_factorization(torch.tensor(self.wide_table.values , dtype = torch.float , requires_grad = False))
        mf.proceed(nfeat = nfeat , print_process = print_process)
        self.Factor = mf
        self.Q = self.Factor.Q
        
    def _IDconvert(self , x):
        try:
            return int(x.decode('utf-8').split('.')[0].split('!')[0])
        except ValueError:
            return -1
        
class dynamic_market_state():
    def __init__(self):
        self.update_day_yield()
        self.update_select_fund()
        
        self.Factor = None
        self.FactorDate = None
        
        self.MarketState = None
        self.MarketStateDate = None
        self.StateSimilarity = None
        
    def update_select_fund(self , select_func = None):
        self._select_func = self._default_select_fund if select_func is None else select_func
            
    def _default_select_fund(self , x):
        # return lambda x:(-x).argpartition((~np.isnan(x)).sum()//100)[:(~np.isnan(x)).sum()//100]
        ns = max((~np.isnan(x)).sum()//100 , 10)
        return (-x).argpartition(ns)[:ns]
            
    def update_day_yield(self , ipo_lag = 31):
        trading_file , info_file = h5py.File(dir_trading , mode = 'r') , h5py.File(dir_info , mode = 'r')
        
        self.SecID , self.TradeDate = np.array(trading_file.get('SecID')[:]) , np.array(trading_file.get('TradeDate')[:])
        cp = np.array(trading_file.get('ClosePrice')[:])
        self.DayYield = np.concatenate((np.tile(np.nan,(cp.shape[0],1)) , cp[:,1:] / cp[:,:-1] - 1) , axis=1)
        
        calendar = pd.DataFrame(info_file.get('calendar')[:]).loc[lambda x:(x.trade > 0)].calendar.to_numpy()
        stock_info = pd.DataFrame(info_file.get('stock_info')[:])
        list_dt = np.tile('20991231',len(self.SecID))
        _s , _i0 , _i1 = np.intersect1d(self.SecID , stock_info.trade_id , return_indices = True)
        list_dt[_i0] = stock_info.list_dt.astype(str)[_i1]

        entry_dt = np.vectorize(lambda x:datetime.strftime(datetime.strptime(x,'%Y%m%d')+timedelta(days = ipo_lag),'%Y%m%d'))(list_dt)
        entry_pos = [(self.TradeDate < ed).sum() for ed in entry_dt.astype(int)]
        
        for si in range(len(self.SecID)): self.DayYield[si,:entry_pos[si]] = np.nan
        
        trading_file.close() , info_file.close()
        assert self.DayYield.shape[:2] == (len(self.SecID) , len(self.TradeDate))
        
    def update_factors(self , SecID , FactorDate , Factor):
        SecID , FactorDate = np.array(SecID) , np.array(FactorDate)
        assert Factor.shape[:2] == (len(SecID) , len(FactorDate))
        _s , _i0 , _i1 = np.intersect1d(self.SecID , SecID , assume_unique=True , return_indices = True)
        
        _FactorDate = FactorDate
        _Factor = np.tile(np.nan , (len(self.SecID) , *Factor.shape[1:]))
        _Factor[_i0] = Factor[_i1]
        
        if (self.FactorDate is not None):
            _FactorDate_old = np.setdiff1d(self.FactorDate,_FactorDate)
            _j0 = np.intersect1d(_FactorDate_old , self.FactorDate , assume_unique=True , return_indices = True)[1]
            _Factor_old = self.Factor[:,_j0]
            
            _FactorDate = np.append(_FactorDate_old , _FactorDate)
            _Factor = np.concatenate((_Factor_old , _Factor) , axis=1)
            
        self.Factor = _Factor[:,np.argsort(_FactorDate)]
        self.FactorDate = _FactorDate[np.argsort(_FactorDate)]
        assert self.Factor.shape[1] == len(self.FactorDate)
    
    def calculate_market_state(self , DateRange):
        self.MarketStateDate = np.array(DateRange)
        self.MarketState = np.tile(np.nan , (len(self.MarketStateDate) , *self.Factor.shape[2:]))
        self.StateSimilarity = np.tile(np.nan , (len(self.SecID) , len(self.MarketStateDate)))
        for i , d in enumerate(self.MarketStateDate):
            _j1 , _j2 = np.where(self.TradeDate == d)[0] , np.where(self.FactorDate <= d)[0]
            if len(_j1) * len(_j2) != 0:
                self.MarketState[i] = np.nanmean(self.Factor[self._select_func(self.DayYield[:,_j1].flatten()),_j2.max()] , axis=0)
                self.StateSimilarity[:,i] = (self.MarketState[i] * self.Factor[:,_j2.max()]).sum(axis=1)
    
    def _cal_mkt_state(self , d):
        _j1 , _j2 = np.where(self.TradeDate == d)[0] , np.where(self.FactorDate <= d)[0]
        if (len(_j1) == 0) or (len(_j2) == 0) :
            return 
        else:
            return np.nanmean(self.Factor[self._select_func(self.DayYield[:,_j1]),_j2.max()] , axis=0)
        
    def _cal_sec_state(self , d):
        _j1 , _j2 = np.where(self.MarketStateDate == d)[0] , np.where(self.FactorDate <= d)[0]
        return (self.MarketState[_j1] * self.Factor[:,_j2.max()]).sum(axis=1)
        
def main():
    t1 = time.time()
    logger.critical('Factor Calculating start!')

    # DynamicStateSimilarity
    dms = dynamic_market_state()
    fs  = fund_stock(SecID = dms.SecID , nfeat = 32 , print_process = False)
    for fd in fs.FactorDate:
        fs.calculate_factors(fd)
        dms.update_factors(fs.SecID , np.array([fd]) , np.expand_dims(fs.Q,1))
        print(f'{fd} factor update done!')
    dms.calculate_market_state(dms.TradeDate[dms.TradeDate > min(dms.FactorDate)])
    save_data_file(f'{dir_data}/Xs_dms.npz' , dms.SecID , dms.MarketStateDate , np.array(['DynamicStateSimilarity']) , dms.StateSimilarity)
    
    t2 = time.time()
    logger.critical('Factor Calculating Finished! Cost {:.2f} Seconds'.format(t2-t1))