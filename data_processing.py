import gc , psutil
import time , argparse
from scripts.data_util.ModelData import (
     DataBlock ,block_process ,block_hist_norm,
     path_block_data,path_norm_data,block_mask,
     block_load_DB,save_dict_data)
from scripts.data_util.DataTank import DataTank
from scripts.function.basic import *
from scripts.util.environ import get_logger , DIR_data
from scripts.util.basic import timer
 
logger = get_logger()
DIR_block     = f'{DIR_data}/block_data'
DIR_hist_norm = f'{DIR_data}/hist_norm'

_save_start_dt , _save_end_dt = 20070101 , None
general_param = {
    'start_dt' : None , 'end_dt' : None , 'mask' : True ,
} # 'start_dt' : 20150101 , 'end_dt' : 20150331 , 'mask' : True ,
process_param = {
    'y' : {
        'DB_source'  : {'labels': {'inner_path' : ['10days/lag1' , '20days/lag1']}} , 
    },
    'trade_day' : {
        'DB_source'  : {
            'trade_day': {            
                'db' : None ,  'inner_path':'day/trade' ,
                'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'vwap' , 'turn_fl'],
            }
        } , 
    },
    'trade_30m' : {
        'DB_source'  : {
            'trade_30m': {
                'db' : 'trade_Xmin' ,  'inner_path':'30min/trade' , 
                'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
            },
            'trade_day': {
                'db' : None ,  'inner_path':'day/trade' ,
                'feature' : ['volume' , 'turn_fl' , 'preclose'] ,
            }
        } ,
    } , 
    # 'gp' : {}
}
"""
process_param = {
    'y' : {
        'DB_source'  : {'labels': {'inner_path' : ['10days/lag1' , '20days/lag1']}} , 
    },
    'trade_day' : {
        'DB_source'  : {
            'trade_day': {            
                'db' : None ,  'inner_path':'day/trade' ,
                'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'vwap' , 'turn_fl'],
            }
        } , 
    },
    'trade_30m' : {
        'DB_source'  : {
            'trade_30m': {
                'db' : 'trade_Xmin' ,  'inner_path':'30min/trade' , 
                'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
            },
            'trade_day': {
                'db' : None ,  'inner_path':'day/trade' ,
                'feature' : ['volume' , 'turn_fl' , 'preclose'] ,
            }
        } ,
    } , 
        'trade_15m' : {
        'DB_source'  : {
            'trade_15m': {
                'db' : 'trade_Xmin' ,  'inner_path':'30min/trade' , 
                'feature' : ['minute' , 'close', 'high', 'low', 'open', 'volume', 'vwap'] ,
            },
            'trade_day': {
                'db' : None ,  'inner_path':'day/trade' ,
                'feature' : ['volume' , 'turn_fl' , 'preclose'] ,
            }
        } ,
    } , 
    'trade_week' : {
        'DB_source'  : {
            'trade_day': {
                'db' : None ,  'inner_path':'day/trade' ,
                'feature' : ['adjfactor', 'preclose' ,'close', 'high', 'low', 'open', 'vwap' , 'turn_fl'],
            },
        } ,
    } , 
    # 'gp' : {}
}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--confirm", type=str, default='')
    try:
        parser = parser.parse_args()
    except:
        parser = parser.parse_args(args=[])
    if parser.confirm == 'no' or not input('You want to update data? print "yes" to confirm!').lower() == 'yes' : exit()

    t1 = time.time()
    logger.critical('Data Processing start!')
    logger.error(f'{len(process_param)} datas :' + str(list(process_param.keys())))

    for key , param in process_param.items():
        tt1 = time.time()
        print(f'{time.ctime()} : {key} start ...')
        
        BlockDict = block_load_DB(param['DB_source'] , **general_param)
        with timer(f'{key} blocks process') as t:
            ThisBlock = block_process(BlockDict , key)
        with timer(f'{key} blocks masking') as t:   
            ThisBlock = block_mask(ThisBlock , **general_param)

        with timer(f'{key} blocks saving ') as t:
            ThisBlock.save(path_block_data(key) , start_dt=_save_start_dt , end_dt=_save_end_dt)

        with timer(f'{key} blocks norming') as t:
            #ThisBlock = DataBlock().read_npz(path_block_data(key))
            block_hist_norm(ThisBlock , key , save_path=path_norm_data(key) , **general_param)
        
        tt2 = time.time()
        print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
    
        del ThisBlock
        gc.collect()

    t2 = time.time()
    logger.critical('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))
