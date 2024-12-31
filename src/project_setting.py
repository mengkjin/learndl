# please check this path before running the code
import sys , socket , torch , yaml
from pathlib import Path

machine_name = socket.gethostname()

# load project setting
setting_path = Path(__file__).absolute().with_suffix('.yaml')
if setting_path.exists():
    with open(setting_path , 'r' , encoding = 'utf-8') as f:
        setting = yaml.load(f , Loader=yaml.FullLoader)

    MY_SERVER = machine_name in setting['SERVERS']
    TERMINAL  = machine_name in setting['TERMINALS']
    MAIN_PATH = Path(setting['MAIN_PATH'][machine_name])
    
else:
    MY_SERVER = machine_name == 'mengkjin-server'
    TERMINAL  = machine_name == 'HNO-JINMENG01'
    if TERMINAL:
        MAIN_PATH = Path('D:/Coding/learndl/learndl')
    elif MY_SERVER:
        MAIN_PATH = Path('/home/mengkjin/workspace/learndl')
    else:
        raise Exception(f'unidentified machine: {machine_name}')

assert MAIN_PATH , f'MAIN_PATH not set for {machine_name}'
assert MY_SERVER or TERMINAL , f'unidentified machine: {machine_name}'
assert (not MY_SERVER) or torch.cuda.is_available() , f'server should have cuda'

assert Path(__file__).is_relative_to(MAIN_PATH) , f'{__file__} is not in {MAIN_PATH}'
sys.path.append(str(MAIN_PATH))

JS_FACTOR_DESTINATION = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if machine_name.lower().startswith(('hno' , 'hpo')) else None