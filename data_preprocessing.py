import gc
import time , argparse
import scripts.util as U
from scripts.util.basic import Timer
from scripts.environ import DIR_data
 
logger = U.logger.get_logger()
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
    'trade_15m' : {
        'DB_source'  : {
            'trade_15m': {
                'db' : 'trade_Xmin' ,  'inner_path':'15min/trade' , 
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

"""

def main(confirm = -1 , parser = None):
    if parser is None:
        parser = argparse.ArgumentParser(description='manual to this script')
        parser.add_argument("--confirm", type=str, default = confirm)
        args , _ = parser.parse_known_args()

    if not args.confirm or not input('Confirm update data? print "yes" to confirm!').lower()[0] == 'y' : exit()

    t1 = time.time()
    logger.critical('Data Processing start!')
    logger.error(f'{len(process_param)} datas :' + str(list(process_param.keys())))

    for key , param in process_param.items():
        tt1 = time.time()
        print(f'{time.ctime()} : {key} start ...')
        
        BlockDict = U.data.ModelData.block_load_DB(param['DB_source'] , **general_param)
        with Timer(f'{key} blocks process'):
            ThisBlock = U.data.ModelData.block_process(BlockDict , key)

        with Timer(f'{key} blocks masking'):   
            ThisBlock = U.data.ModelData.block_mask(ThisBlock , **general_param)

        with Timer(f'{key} blocks saving '):
            ThisBlock.save(U.data.ModelData.path_block_data(key) , start_dt=_save_start_dt , end_dt=_save_end_dt)

        with Timer(f'{key} blocks norming'):
            #ThisBlock = U.data.ModelData.DataBlock().read_npz(U.data.ModelData.path_block_data(key))
            U.data.ModelData.block_hist_norm(ThisBlock , key , save_path=U.data.ModelData.path_norm_data(key) , **general_param)
        
        tt2 = time.time()
        print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
    
        del ThisBlock
        gc.collect()

    t2 = time.time()
    logger.critical('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))


if __name__ == '__main__':
    main()