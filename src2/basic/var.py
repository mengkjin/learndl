import torch

THIS_IS_SERVER = torch.cuda.is_available() # socket.gethostname() == 'mengkjin-server'
assert not THIS_IS_SERVER or torch.cuda.is_available() , f'SERVER must have cuda available'

