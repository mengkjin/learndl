import os , shutil , socket , torch , yaml
from dataclasses import dataclass
from typing import Literal , Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SRC_DIR)
BOOSTER_MODULE = ['lgbm']
THIS_IS_SERVER = socket.gethostname() == 'mengkjin-server'
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'mengkjin-server must have cuda available'

@dataclass
class _CustomPath:
    main        : str = MAIN_DIR
    data        : str = os.path.join(MAIN_DIR , 'data')
    batch       : str = os.path.join(MAIN_DIR , 'data' , 'MiniBatch')
    block       : str = os.path.join(MAIN_DIR , 'data' , 'DataBlock')
    database    : str = os.path.join(MAIN_DIR , 'data' , 'DataBase')
    dataset     : str = os.path.join(MAIN_DIR , 'data' , 'DataSet')
    norm        : str = os.path.join(MAIN_DIR , 'data' , 'HistNorm')
    tree        : str = os.path.join(MAIN_DIR , 'data' , 'TreeData')
    updater     : str = os.path.join(MAIN_DIR , 'data' , 'Updater')
    conf        : str = os.path.join(MAIN_DIR , 'configs')
    logs        : str = os.path.join(MAIN_DIR , 'logs')
    model       : str = os.path.join(MAIN_DIR , 'model')
    result      : str = os.path.join(MAIN_DIR , 'result')

    def __post_init__(self):
        [os.makedirs(v , exist_ok=True) for v in self.__dict__.values()]

    @staticmethod
    def read_yaml(yaml_file):
        with open(yaml_file ,'r') as f:
            d = yaml.load(f , Loader = yaml.FullLoader)
        return d
    
    @staticmethod
    def dump_yaml(data , yaml_file):
        with open(yaml_file , 'a' if os.path.exists(yaml_file) else 'w') as f:
            yaml.dump(data , f)

    @staticmethod
    def copytree(src , dst):
        shutil.copytree(src , dst)

    @staticmethod
    def copyfiles(src , dst , bases):
        [shutil.copyfile(f'{src}/{base}' , f'{dst}/{base}') for base in bases]

    @staticmethod
    def deltrees(dir , bases):
        [shutil.rmtree(f'{dir}/{base}') for base in bases]

@dataclass
class _CustomConf:
    SILENT        : bool = False
    SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
    SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
    SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
    SAVE_OPT_MODEL: Literal['pt'] = 'pt'

@dataclass(slots=True)
class _RegModel:
    name : str
    type : Literal['best' , 'swalast' , 'swabest']
    num  : int
    alias : Optional[str] = None

PATH = _CustomPath()
CONF = _CustomConf()
REG_MODELS = [
    _RegModel('gru_day'    , 'swalast' , 0 , 'gru_day_V0') ,
    _RegModel('gruRTN_day' , 'swalast' , 0 , 'gruRTN_day_V0') , 
    _RegModel('gruRES_day' , 'swalast' , 0 , 'gruRES_day_V0') ,
]

RISK_STYLE = ['size','beta','momentum','residual_volatility','non_linear_size',
                'book_to_price','liquidity','earnings_yield','growth','leverage']
