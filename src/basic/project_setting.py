# please check this path before running the code
import sys , socket , torch
from pathlib import Path
from typing import Any

# variables
#   machine types , will decide path and update process of data and models
IS_SERVER = torch.cuda.is_available() 
MY_SERVER = socket.gethostname() == 'mengkjin-server'
MY_LAPTOP = socket.gethostname() == 'HNO-JINMENG01'

# main path , depending on machine types
#   please check this path before running the code
if MY_LAPTOP:
    MAIN_PATH = Path('D:/Coding/learndl/learndl')
elif MY_SERVER:
    MAIN_PATH = Path('/home/mengkjin/Workspace/learndl')
else:
    raise Exception(f'unidentified machine: {socket.gethostname()}')
sys.path.append(str(MAIN_PATH))

# if the machine is in js environment , set the (additional) path for factor storage
JS_ENVIRONMENT = socket.gethostname().lower().startswith(('hno' , 'hpo'))
JS_FACTOR_DESTINATION = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha') if JS_ENVIRONMENT else None

INSTANCE_RECORD : dict[str, Any] = {}
