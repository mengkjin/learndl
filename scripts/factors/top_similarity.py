import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py , time , traceback
from ..data_util.ModelData import DataBlock
from ..data_util.DataTank import DataTank
from ..util.environ import get_logger,DIR_data
from ..util.trainer import Device
from ..function.basic import *
from datetime import datetime,timedelta

logger = get_logger()
path_port  = f'{DIR_data}/fund_stock_port.h5'
path_trade = f'{DIR_data}/DB_trade_day.h5'
path_info  = f'{DIR_data}/DB_information.h5'
save_path = f'{DIR_data}/block_data/X_top_similarity.npz'

class matrix_factorization():
    def __init__(self , m , learn_rates = [0.1,0.05,0.01,0.005,0.001]):
        self.device = Device()
        self.mat  = self.device(m)
        self.nrow = m.shape[0]
        self.ncol = m.shape[1]
        self.learn_rates = learn_rates
    
    def proceed(self , nfeat , ltheta = 0.01 , num_epochs = 200 , print_process = False):
        self.mod = []
        self.losses = []
        for learn_rate in self.learn_rates:
            mod = self.device(self.factor_module((self.nrow , self.ncol , nfeat)))
            criterion = nn.MSELoss(reduction = 'sum')
            optimizer = torch.optim.Adam(mod.parameters(), lr = learn_rate) #type: ignore

            losses = []  
            for epoch in range(num_epochs):
                output , theta = mod() # type: ignore
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
    def __init__(self , nfeat = 32 , print_process = False , default_fund_id = None , default_secid = None , 
                 start_dt = None , end_dt = None):
        self.port_file = path_port
        with h5py.File(self.port_file , mode='r') as file:
            self.port_date = np.array(list(file.keys())).astype(int)
        if start_dt is not None: self.port_date = self.port_date[self.port_date >= start_dt]
        if end_dt   is not None: self.port_date = self.port_date[self.port_date <= end_dt]
        self.fund_id = default_fund_id
        self.secid = default_secid
        self.nfeat = nfeat
        self.print_process = print_process

    def load_port(self , date):
        with h5py.File(self.port_file , mode='r') as file:
            tb = pd.DataFrame(file[str(date)][:]) #type: ignore

        tb.columns = ['fund_id','secid','weight']
        tb.fund_id = [self._IDconvert(s) for s in tb.fund_id]
        tb.secid   = [self._IDconvert(s) for s in tb.secid]
        return tb
        
    def calculate_factors(self , date):
        tb = self.load_port(date)
        if self.fund_id is None:
            target_fund_id = sorted(tb.fund_id[tb.fund_id > 0].unique())  
        else:
            target_fund_id = np.array(self.fund_id)

        if self.secid is None:
            target_secid = sorted(tb.secid[tb.secid > 0].unique())
            self.secid = target_secid
        else:
            target_secid = np.array(self.secid)
        
        tb = tb[tb.fund_id.isin(target_fund_id) & tb.secid.isin(target_secid)]
        tb = pd.concat([pd.DataFrame({'fund_id':target_fund_id,'secid':target_secid[0],'weight':0.,}) ,
                        pd.DataFrame({'fund_id':target_fund_id[0],'secid':target_secid,'weight':0.,}) , 
                        tb, ])
        self.wide_table = tb.pivot_table('weight','fund_id','secid',aggfunc='sum',fill_value=0.)
        self.wide_table = self.wide_table.loc[target_fund_id,target_secid] #type: ignore
        assert np.array_equal(self.wide_table.columns.tolist() , target_secid)
        assert np.array_equal(self.wide_table.index.tolist() , target_fund_id)
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
    def __init__(self , start_dt = None , end_dt = None):
        self.start_dt = 20140430 if start_dt is None else start_dt
        self.end_dt   = 99991231 if end_dt   is None else end_dt
        
        self.port_factors = None
        self.port_date = None
        
        self.StateSimilarity = None

        self.load_day_yield()
        self.update_top_sec()
        
    def load_day_yield(self , ipo_lag = 31):
        __start_time__ = time.time()
        dtank_info = DataTank(path_info , open = True , mode = 'r')
        dtank_trade = DataTank(path_trade , open = True , mode = 'r')
        try:
            self.calendar   = dtank_info.read_dataframe('/basic/calendar')
            self.stock_info = dtank_info.read_dataframe('/stock/description')

            trade_calendar = self.calendar.loc[lambda x:(x.trade > 0)].calendar.to_numpy()
            trade_calendar = trade_calendar[(trade_calendar >= self.start_dt) * (trade_calendar <= self.end_dt) > 0]

            valid_date = np.array(list(dtank_trade.get_object('/day/trade').keys())).astype(int) #type:ignore
            valid_date = np.intersect1d(valid_date , trade_calendar)
            df_list = {str(date):dtank_trade.read_data1D(f'/day/trade/{date}',feature='pctchange').to_dataframe().reset_index() for date in valid_date}  #type:ignore
            for date , df in df_list.items(): df.insert(0,'date',int(date))
            df = pd.concat(df_list.values())
            df['pctchange'] = df['pctchange'] / 100
            df = df.pivot_table('pctchange','secid','date')
            self.secid , self.trade_date = df.index.values.astype(int) , df.columns.values
            self.stock_info = pd.concat([self.stock_info , 
                                         pd.DataFrame({'secid':np.setdiff1d(self.secid , self.stock_info.secid)})])

            self.day_yield = df.values
            list_dt = self.stock_info.set_index('secid').loc[self.secid,'list_dt'].fillna(21991231).astype(int).astype(str)
            valid_date_func = lambda x:datetime.strftime(datetime.strptime(x,'%Y%m%d')+timedelta(days = ipo_lag),'%Y%m%d')
            entry_pos = [(self.trade_date < d).sum() for d in np.vectorize(valid_date_func)(list_dt).astype(int)]
            for i in range(len(self.secid)): self.day_yield[i,:entry_pos[i]] = np.nan
            
            assert self.day_yield.shape[:2] == (len(self.secid) , len(self.trade_date))
            print(f'loading: {time.time() - __start_time__:.2f} secs')
        except:
            traceback.print_exc()
        finally:
            dtank_info.close() 
            dtank_trade.close()
    
    def update_top_sec(self , func = None):
        self.top_sec = self._default_top_sec if func is None else func
            
    def _default_top_sec(self , x):
        # return lambda x:(-x).argpartition((~np.isnan(x)).sum()//100)[:(~np.isnan(x)).sum()//100]
        ns = max((~np.isnan(x)).sum()//100 , 10)
        return (-x).argpartition(ns)[:ns]
        
    def update_factors(self , secid , port_date , port_factors):
        if not isinstance(port_date,(list,tuple,np.ndarray)): port_date = [port_date]
        secid , port_date = np.array(secid) , np.array(port_date)
        assert port_factors.shape[:2] == (len(secid) , len(port_date))
        _ , int_i0 , int_i1 = np.intersect1d(self.secid , secid , assume_unique=True , return_indices = True)
        
        new_port_date = port_date
        new_port_factors = np.tile(np.nan , (len(self.secid) , *port_factors.shape[1:]))
        new_port_factors[int_i0] = port_factors[int_i1]
        
        if (self.port_date is not None):
            old_port_date = self.port_date[np.isin(self.port_date , port_date)]
            old_port_factors = self.port_factors[:,np.isin(self.port_date , port_date)]  #type:ignore
            
            new_port_date = np.append(old_port_date , new_port_date)
            new_port_factors = np.concatenate((old_port_factors , new_port_factors) , axis=1)
            
        self.port_factors = new_port_factors[:,np.argsort(new_port_date)]
        self.port_date = new_port_date[np.argsort(new_port_date)]
        assert self.port_factors.shape[1] == len(self.port_date)
    
    def calculate_market_state(self , start_dt = None , end_dt = None):
        date_range = self.trade_date
        if start_dt is not None: date_range = date_range[date_range >= start_dt]
        if end_dt   is not None: date_range = date_range[date_range <= end_dt]
        
        self.market_top_state = np.tile(np.nan , (len(date_range) , *self.port_factors.shape[2:]))  #type:ignore
        self.top_similarity = np.tile(np.nan , (len(self.secid) , len(date_range)))
        for i , d in enumerate(date_range):
            j1 , j2 = np.where(self.trade_date == d)[0] , np.where(self.port_date <= d)[0]
            if len(j1) * len(j2) != 0:
                top_sec = self.top_sec(self.day_yield[:,j1].flatten())
                self.market_top_state[i] = np.nanmean(self.port_factors[top_sec][:,j2.max()],axis=0)  #type:ignore
                self.top_similarity[:,i] = (self.market_top_state[i] * self.port_factors[:,j2.max()]).sum(axis=1)  #type:ignore
        self.state_date = date_range
        
def main(start_dt = None , end_dt = None):
    t1 = time.time()
    logger.critical('top_similarity Factor Calculating start!')

    # DynamicStateSimilarity : top_similarity
    fs  = fund_stock(nfeat = 32 , print_process = False , start_dt = start_dt , end_dt = end_dt)
    start_dt = fs.port_date[0] if start_dt is None else max(fs.port_date[0] , start_dt)
    dms = dynamic_market_state(start_dt=start_dt , end_dt = end_dt)
    for port_date in fs.port_date:
        fs.calculate_factors(port_date)
        dms.update_factors(fs.secid , port_date , np.expand_dims(fs.Q,1))
        print(f'{port_date} factor update done!')
    dms.calculate_market_state(start_dt=start_dt , end_dt = end_dt)
    values = dms.top_similarity if len(dms.top_similarity.shape) == 3 else dms.top_similarity[:,:,None]
    block = DataBlock(values , dms.secid , dms.state_date , 'top_similarity')
    block.save(save_path)
    
    t2 = time.time()
    logger.critical('top_similarity Factor Calculating Finished! Cost {:.2f} Seconds'.format(t2-t1))