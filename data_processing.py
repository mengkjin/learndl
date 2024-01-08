import gc , psutil
import time , argparse
from scripts.data_util.ModelData import (
     DataBlock ,block_process ,block_hist_norm,
     path_block_data,path_norm_data,block_mask)
from scripts.data_util.DataTank import DataTank
from scripts.function.basic import *
from scripts.util.environ import get_logger , DIR_data
from scripts.util.basic import timer
 
logger = get_logger()
DIR_block  = f'{DIR_data}/block_data'
DIR_hist_norm = f'{DIR_data}/hist_norm'

_save_start_dt , _save_end_dt = 20070101 , None
general_param = {
    'start_dt' : None , 'end_dt' : None , 'mask' : True ,
} # 'start_dt' : 20150101 , 'end_dt' : 20150331 , 'mask' : True ,
process_param = {
    'y' : {
        'DB_key'  : 'labels' , 
        'DB_path' : ['10days/lag1' , '20days/lag1'] ,
    },
    'trade_day' : {
        'DB_key'  : 'trade_day' , 
        'DB_path' : 'day/trade' ,
        'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'volume', 'vwap'] , 'process_method' : 'adj_order' ,
    },
    'trade_15m' : {
        'DB_key'  : 'trade_Xmin' ,
        'DB_path' : '15min/trade' ,
        'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] , 'process_method' : 'order' ,
    },
    # 'gp' : {}
}
"""
    'y' : {
        'DB_key'  : 'labels' , 
        'DB_path' : ['10days/lag1' , '20days/lag1'] ,
    },
    'trade_day' : {
        'DB_key'  : 'trade_day' , 
        'DB_path' : 'day/trade' ,
        'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'volume', 'vwap'] , 'process_method' : 'adj_order' ,
    },
    'trade_15m' : {
        'DB_key'  : 'trade_Xmin' ,
        'DB_path' : '15min/trade' ,
        'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] , 'process_method' : 'order' ,
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

            for key , param in process_param.items():
                tt1 = time.time()
                print(f'{time.ctime()} : {key} start ...')
                
                blocks = []
                db_key = param['DB_key']
                db_path = [param['DB_path']] if isinstance(param['DB_path'] , str) else param['DB_path']
                kwargs = {**param , **general_param}

                for path in db_path:
                    with timer(f'{key} blocks reading {path} Data1D\'s') as t:
                        blocks.append(DataBlock().from_db(db_key , path , **kwargs))

                with timer(f'{key} blocks merging') as t:
                    ThisBlock = DataBlock().merge_others(blocks)
                    del blocks
                    gc.collect()

                with timer(f'{key} blocks process') as t:
                    ThisBlock = block_mask(ThisBlock , **kwargs)
                    ThisBlock = block_process(ThisBlock , **kwargs)

                with timer(f'{key} blocks saving ') as t:
                    ThisBlock.save(path_block_data(key) , start_dt=_save_start_dt , end_dt=_save_end_dt)
                
                with timer(f'{key} blocks norming') as t:
                    #ThisBlock = DataBlock().read_npz(path_block_data(key))
                    block_hist_norm(ThisBlock , key , path_norm_data(key) , **param)
                
                tt2 = time.time()
                print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
            
                del ThisBlock
                gc.collect()

            t2 = time.time()
            logger.critical('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))
