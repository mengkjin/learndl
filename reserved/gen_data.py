import torch , h5py
import numpy as np
import pandas as pd
import os, shutil , gc , time , argparse
from learndl.scripts.function.basic import *
from ..scripts.util.environ import get_logger , get_config

NBARS      = {'day' : 1 , '15m' : 16 ,}
BEFORE_DAY = 20170101
STEP_DAY   = 5
DATATYPE   = get_config('data_type')['DATATYPE']


update_files = ['day_trading_data' , 'day_ylabels_data' , '15m_trading_data']
data_index_dict = {'day' : ('SecID' , 'TradeDate') , '15m' : ('SecID' , 'TradeDateTime') , '30m' : ('SecID' , 'TradeDateTime') ,
                   'gp' : ('SecID' , 'TradeDate') ,}

dir_nas    = None # f'/root/autodl-nas'
dir_data   = f'./data'
dir_update = f'{dir_data}/update_data'

path_ydata = f'{dir_data}/Ys.npz'
path_xdata = lambda x:f'{dir_data}/Xs_{x}.npz'
path_norm_param = f'{dir_data}/norm_param.pt'

logger = get_logger()

def fetch_update_from_nas():
    source_dir = dir_nas
    if source_dir is None: return
    target_dir = dir_update
    os.makedirs(target_dir , exist_ok=True)
    fetch_list = []
    for file_starter in update_files:
        f_list = [f for f in os.listdir(source_dir) if f.startswith(file_starter)]
        for f in f_list:
            shutil.copy(f'{source_dir}/{f}', f'{target_dir}/{f}')
            os.remove(f'{source_dir}/{f}')
            fetch_list.append(f)
    if len(fetch_list) > 0 : print('{:s} copy file finished!'.format(', '.join(fetch_list)))
    return

def update_trading_data(remove_update_file = True):
    fetch_update_from_nas()
    target_dir = dir_data
    source_dir = dir_update
    for file_starter in update_files:
        row_var , col_var = data_index_dict[file_starter.split('_')[0]]
        
        target_path = f'{target_dir}/{file_starter}.h5'
        source_path = sorted([f'{source_dir}/{f}' for f in os.listdir(source_dir) if f.startswith(file_starter)])
        if len(source_path) > 0 and os.path.exists(target_path) == 0:
            shutil.copy(source_path[0] , target_path)
            source_path = source_path[1:]
        if len(source_path) == 0: continue
        target_file = h5py.File(target_path , mode='r+')
        source_file = [h5py.File(f , mode='r') for f in source_path]

        row_tuple = tuple([f.get(row_var)[:] for f in [target_file] + source_file])
        col_tuple = tuple([f.get(col_var)[:] for f in [target_file] + source_file])
        row_all , col_all = None , None
        
        for k in sorted(list(target_file.keys() - [row_var , col_var])):
            t0 = time.time()
            data = tuple([f.get(k)[:] for f in [target_file] + source_file])
            data , row_all , col_all = merge_data_2d(data , row_tuple , col_tuple , row_all , col_all)
            row_all , col_all = np.array(row_all).astype(int) , np.array(col_all).astype(int)
            
            del target_file[k]
            target_file.create_dataset(k , data = data , compression="gzip")
            print(f'{file_starter} -> {k} cost {(time.time() - t0):.2f}')
        
        del target_file[row_var]
        target_file.create_dataset(row_var , data = row_all , compression="gzip")

        del target_file[col_var]
        target_file.create_dataset(col_var , data = col_all , compression="gzip")

        [f.close() for f in source_file]
        target_file.close()
        if remove_update_file: [os.remove(f'{source_dir}/{f}') for f in os.listdir(source_dir) if f.startswith(file_starter)]
        print(f'Update {file_starter} Finished! From {min(col_all)} to {max(col_all)} , of {len(row_all)} stocks')
    return

def prepare_model_data():
    source_dir = dir_data
    target_dir = dir_data

    for file_starter in update_files:
        print(f'Preparing {file_starter} data...')
        model_data_type , feature_type = file_starter.split('_')[0] , file_starter.split('_')[1]
        row_var , col_var = data_index_dict[model_data_type]
        
        source_file = h5py.File(f'{source_dir}/{file_starter}.h5' , mode = 'r')
        row , col = source_file.get(row_var)[:] , source_file.get(col_var)[:]
        
        if feature_type == 'ylabels':
            feat = ['Y10Delay' , 'Y5Delay']
            file_path = path_ydata
        elif feature_type == 'trading':
            feat = ['OpenPrice','HighPrice','LowPrice','ClosePrice','TradeVolume','VWPrice']
            file_path = path_xdata(model_data_type)
        else:
            raise Exception(f'KeyError : {feature_type}')
            
        arr = np.array([source_file.get(k)[:] for k in feat]).transpose(1,2,0)
        
        if col_var == 'TradeDateTime':
            col = col // 100
            assert sum([j != NBARS[model_data_type] for j in [list(col).count(i) for i in set(col)]]) == 0
            col = col[::NBARS[model_data_type]]
            arr = arr.reshape(arr.shape[0] , -1 , NBARS[model_data_type] , arr.shape[2])
        else:
            arr = arr.reshape(arr.shape[0] , -1 , 1 , arr.shape[2])
        
        assert(arr.shape[0] , arr.shape[1]) == (len(row) , len(col))
        
        save_data_file(file_path , row , col , feat , arr)
        source_file.close()
        print(f'arr shape : {arr.shape} , row shape : {row.shape} , col shape : {col.shape}')
    return

def save_data_file(file_path , row , col , feat , arr):
    if len(arr.shape) == 3:
        arr = arr.reshape(arr.shape[0],arr.shape[1],1,arr.shape[3])
    elif len(arr.shape) == 2:
        arr = arr.reshape(arr.shape[0],arr.shape[1],1,1)
    elif len(arr.shape) == 1:
        raise Exception(f'DimError: shape is {str(arr.shape)}')
    assert (arr.shape[0] , arr.shape[1] , arr.shape[-1]) == (len(row) , len(col) , len(feat))
    np.savez_compressed(file_path , row = row , col = col , feat = feat , arr = arr)

def cal_norm_param(maxday = 60 , before_day = BEFORE_DAY , step_day = STEP_DAY):
    norm_param = {}
    for model_data_type in DATATYPE['trade']:
        if not os.path.exists(path_xdata(model_data_type)): continue
        logger.error(f'[{model_data_type}] Data avg and std generation start!')
        t0 = time.time()
        x_dict = np.load(path_xdata(model_data_type))
        
        row_data , col_data = np.array(x_dict['row'] , dtype = int) , np.array(x_dict['col'] , dtype = int)
        beg_col_id = (col_data < before_day).sum()
        x = torch.tensor(np.array(x_dict['arr'])[:, :beg_col_id, :]).to(dtype = torch.float)
        
        print(f'Loading {model_data_type} trading data finished, cost {time.time() - t0:.2f} Secs')
        stock_n , day_len , _ , feat_dim = x.shape
        step_len = day_len // step_day
        bars_len = maxday * NBARS[model_data_type]
        padd_len = (0,0,0,0,0,max(0 , maxday - step_day),0,0)
        
        x = torch.nn.functional.pad(x,padd_len,value=np.nan)
        avg_x = torch.zeros(bars_len , feat_dim)
        std_x = torch.zeros(bars_len , feat_dim)
        
        x_div = torch.ones(stock_n , step_len , 1 , feat_dim)
        x_div.copy_(x[:,(maxday - 1):(maxday - 1 + day_len):step_day,-1:])
        print(x_div.shape)
        
        nan_sample = (x_div == 0).sum(dim = (-2,-1)) > 0
        for i in range(maxday):
            nan_sample += x[:,i:(i+day_len):step_day,:,:].reshape(stock_n,step_len,-1).isnan().any(dim = -1)

        for i in range(maxday):
            # (stock_n , step_len)(nonnan_sample) , day_bars , feat_dim
            vijs = (x[:,i:(i+day_len):step_day,:,:] / x_div)[nan_sample == 0]
            avg_x[i*NBARS[model_data_type]:(i+1)*NBARS[model_data_type]] = vijs.mean(dim = 0)
            std_x[i*NBARS[model_data_type]:(i+1)*NBARS[model_data_type]] = vijs.std(dim = 0)
        assert avg_x.isnan().sum() + std_x.isnan().sum() == 0

        norm_param.update({model_data_type : {'avg' : avg_x , 'std' : std_x}})
        del x
        gc.collect()
        
    torch.save(norm_param , path_norm_param)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--confirm", type=str, default='')
    if parser.parse_args().confirm == 'no':
        pass
    else:
        a = input('You want to update data? print "yes" to confirm!')
        if a == 'yes':
            t1 = time.time()
            logger.critical('Data loading start!')

            update_trading_data()
            prepare_model_data()
            cal_norm_param()
            
            t2 = time.time()
            logger.critical('Data loading Finished! Cost {:.2f} Seconds'.format(t2-t1))