import torch
from ..classes import BatchData

def batch_data(batch_size = 2 , seq_len = 30 , n_inputs = 6 , predict_steps = 1):
    patch_len = 3
    stride = 2
    mask_ratio = 0.4
    d_model = 16

    x = torch.rand(batch_size , seq_len , n_inputs)

    y = torch.rand(batch_size , predict_steps)
    w = None
    i = torch.Tensor([])
    v = torch.Tensor([])
    return BatchData(x , y , w , i , v)



