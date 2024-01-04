import gc , psutil
import time , argparse
from scripts.data_util.ModelData import (
     DataBlock ,block_process ,block_hist_norm,path_block_data,path_norm_data)
from scripts.data_util.DataTank import DataTank
from scripts.function.basic import *
from scripts.util.environ import get_logger , DIR_data
from scripts.util.basic import timer
 
logger = get_logger()
DIR_block  = f'{DIR_data}/block_data'
DIR_hist_norm = f'{DIR_data}/hist_norm'

_load_start_dt , _load_end_dt = None , None # 20150101 , 20150331 # None , None
_save_start_dt , _save_end_dt = 20070101 , None
process_param = {
    'y' : {
        '__path_DB__' : 'DB_labels.h5' ,
        '__path_data__' : ['10days/lag1' , '20days/lag1'] ,
        'start_dt' : _load_start_dt , 'end_dt' : _load_end_dt
    },
    'trade_day' : {
        '__path_DB__' : 'DB_trade_day.h5' ,
        '__path_data__' : 'day/trade' ,
        'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
        'process_method' : 'adj_order' ,
        'start_dt' : _load_start_dt , 'end_dt' : _load_end_dt
    },
    'trade_15m' : {
        '__path_DB__' : 'DB_trade_Xmin.h5' ,
        '__path_data__' : '15min/trade' ,
        'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
        'process_method' : 'order' ,
        'start_dt' : _load_start_dt , 'end_dt' : _load_end_dt
    },
    # 'gp' : {}
}
"""
    'y' : {
        '__path_DB__' : 'DB_labels.h5' ,
        '__path_data__' : ['10days/lag1' , '20days/lag1'] ,
        'start_dt' : _load_start_dt , 'end_dt' : _load_end_dt
    },
    'trade_day' : {
        '__path_DB__' : 'DB_trade_day.h5' ,
        '__path_data__' : 'day/trade' ,
        'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
        'process_method' : 'adj_order' ,
        'start_dt' : _load_start_dt , 'end_dt' : _load_end_dt
    },
    'trade_15m' : {
        '__path_DB__' : 'DB_trade_Xmin.h5' ,
        '__path_data__' : '15min/trade' ,
        'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
        'process_method' : 'order' ,
        'start_dt' : _load_start_dt , 'end_dt' : _load_end_dt
    },
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--confirm", type=str, default='')
    if parser.parse_args().confirm == 'no':
        pass
    else:
        a = input('You want to update data? print "yes" to confirm!')
        if a == 'yes':
            t1 = time.time()
            logger.critical('Data Processing start!')
            logger.error(f'{len(process_param)} datas :' + str(list(process_param.keys())))
            x_trade_norm_dict = dict()
            for key , param in process_param.items():
                tt1 = time.time()
                print(f'{time.ctime()} : {key} start ...')
                
                blocks = []
                dtank  = DataTank('/'.join([DIR_data,param['__path_DB__']]) , 'r')
                if isinstance(param['__path_data__'] , str): param['__path_data__'] = [param['__path_data__']]
                for f in param['__path_data__']:
                    with timer(f'{key} blocks reading {f} Data1D\'s') as t:
                        blocks.append(DataBlock().from_dtank(dtank,f,**param))
                dtank.close()

                with timer(f'{key} blocks merging') as t:
                    ThisBlock = DataBlock().merge_others(blocks)
                    del blocks
                    gc.collect()

                with timer(f'{key} blocks process') as t:
                    ThisBlock = block_process(ThisBlock , **param)

                with timer(f'{key} blocks saving ') as t:
                    ThisBlock.save(path_block_data(key) , start_dt=_save_start_dt , end_dt=_save_end_dt)
                
                with timer(f'{key} blocks norming') as t:
                    #del ThisBlock
                    #ThisBlock = DataBlock().read_npz(path_block_data(key))
                    block_hist_norm(ThisBlock , key , path_norm_data(key) , **param)
                
                tt2 = time.time()
                print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
            
                del ThisBlock , dtank
                gc.collect()

            t2 = time.time()
            logger.critical('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))
