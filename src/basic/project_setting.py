import sys , socket , torch
from pathlib import Path
from typing import Any ,Literal

# variables
THIS_IS_SERVER  = torch.cuda.is_available() and socket.gethostname() == 'mengkjin-server'
MAIN_PATH = Path('D:/Coding/learndl/learndl') if not THIS_IS_SERVER else Path('/home/mengkjin/Workspace/learndl')
sys.path.append(str(MAIN_PATH))

INSTANCE_RECORD : dict[Literal['trainer'] , Any] = {}