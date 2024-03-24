import gc , sys , time , argparse
import src.util as U

from datetime import datetime , timedelta
from dataclasses import dataclass , field

from src.util.basic import Timer
from src.util.logger import Logger
from src.environ import DIR_data
from src.data.BlockData import DataBlock, blocks_process
 
logger = Logger()

DIR_block     = f'{DIR_data}/block_data'
DIR_hist_norm = f'{DIR_data}/hist_norm'


def today(offset = 0):
    d = datetime.today() + timedelta(days=offset)
    return int(d.strftime('%Y%m%d'))

@dataclass
class DataProcessConfig:
    load_start_dt : int | None
    load_end_dt   : int | None
    save_start_dt : int | None
    save_end_dt   : int | None
    hist_start_dt : int | None
    hist_end_dt   : int | None
    mask : dict = field(default_factory=dict)
    data : dict = field(default_factory=dict)

Configs = DataProcessConfig(
    load_start_dt = None ,
    load_end_dt   = None ,
    save_start_dt = 20070101 ,
    save_end_dt   = None ,
    hist_start_dt = None ,
    hist_end_dt   = 20161231 ,  
    mask          = {'list_dt':True}
)
Configs.data['y'] = {
    'DB_source'  : {
        'labels': {'inner_path' : ['10days/lag1' , '20days/lag1']} ,
        'models': {'inner_path' : 'risk_model/exposure'} ,
    }
}
Configs.data['trade_day'] = {
    'DB_source'  : {
        'trade_day': {   
            'db' : None ,  
            'inner_path':'day/trade' ,
            'feature' : ['adjfactor', 'close', 'high', 'low', 'open', 'vwap' , 'turn_fl'],
        }
    } , 
}
Configs.data['trade_30m'] = {
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
}
"""
Configs.data['trade_15m'] = {
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
}
Configs.data['trade_week'] = {
    'DB_source'  : {
        'trade_day': {
            'db' : None ,  'inner_path':'day/trade' ,
            'feature' : ['adjfactor', 'preclose' ,'close', 'high', 'low', 'open', 'vwap' , 'turn_fl'],
        },
    } ,

}
"""

def main(if_train = True, confirm = 0 , parser = None):
    if parser is None:
        parser = argparse.ArgumentParser(description='manual to this script')
        parser.add_argument("--confirm", type=str, default = confirm)
        args , _ = parser.parse_known_args()

    if if_train:
        if not args.confirm and not input('Confirm update data? print "yes" to confirm!').lower()[0] == 'y' : 
            sys.exit()

    t1 = time.time()
    logger.critical(f'if_train is {if_train} , Data Processing start!')
    logger.error(f'{len(Configs.data)} datas :' + str(list(Configs.data.keys())))

    for key , param in Configs.data.items():
        tt1 = time.time()
        print(f'{time.ctime()} : {key} start ...')
        
        BlockDict = DataBlock.load_DB_source(
            param['DB_source'] , 
            start_dt = Configs.load_start_dt if if_train else today(-181), 
            end_dt   = Configs.load_end_dt   if if_train else None)
        
        with Timer(f'{key} blocks process'):
            ThisBlock = blocks_process(BlockDict , key)

        with Timer(f'{key} blocks masking'):   
            ThisBlock = ThisBlock.mask_values(mask = Configs.mask)

        with Timer(f'{key} blocks saving '):
            ThisBlock.save(
                key , if_train , 
                start_dt = Configs.save_start_dt if if_train else None , 
                end_dt   = Configs.save_end_dt   if if_train else None)

        with Timer(f'{key} blocks norming'):
            ThisBlock.hist_norm(
                key , if_train ,
                start_dt  = Configs.hist_start_dt , 
                end_dt    = Configs.hist_end_dt)
        
        tt2 = time.time()
        print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
    
        del ThisBlock
        gc.collect()

    t2 = time.time()
    logger.critical('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))


if __name__ == '__main__':
    main(if_train=False)
    main(if_train=True)