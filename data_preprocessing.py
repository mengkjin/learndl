import gc , sys , time , argparse

from src.util import Logger , Timer
from src.data.BlockData import DataBlock , DataProcessConfig
 
logger = Logger()

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

    Configs = DataProcessConfig(if_train,blocks = ['y' , 'trade_day' , 'trade_30m'])
    logger.error(f'{len(Configs.blocks)} datas :' + str(list(Configs.blocks)))

    for key , param in Configs.get_block_params():
        tt1 = time.time()
        print(f'{time.ctime()} : {key} start ...')
        
        BlockDict = DataBlock.load_DB(param , Configs.load_start_dt, Configs.load_end_dt)
        
        with Timer(f'{key} blocks process'):
            ThisBlock = DataBlock.blocks_process(BlockDict , key)

        with Timer(f'{key} blocks masking'):   
            ThisBlock = ThisBlock.mask_values(mask = Configs.mask)

        with Timer(f'{key} blocks saving '):
            ThisBlock.save(key , if_train , Configs.save_start_dt , Configs.save_end_dt)

        with Timer(f'{key} blocks norming'):
            ThisBlock.hist_norm(key , if_train , Configs.hist_start_dt , Configs.hist_end_dt)
        
        tt2 = time.time()
        print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
    
        del ThisBlock
        gc.collect()

    t2 = time.time()
    logger.critical('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))


if __name__ == '__main__':
    main(False)
    main(True)