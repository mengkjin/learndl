import torch

def align_shape(label : torch.Tensor , pred : torch.Tensor , w : torch.Tensor | None = None):
    if label.shape[-1] != pred.shape[-1]:
        last_dim = min(label.shape[-1] , pred.shape[-1])
        label = label[...,:last_dim]
        pred = pred[...,:last_dim]
        if w is not None:
            w = w[...,:last_dim]
    return label , pred , w
