import torch
from ..classes import BaseBuffer , BaseDataModule

class BufferSpace(BaseBuffer):
    '''dynamic buffer space for some module to use (tra), can be updated at each batch / epoch '''
    def register_setup(self) -> None:
        # first param of wrapper is container, which represent self in ModelData
        if self.key == 'tra':
            def tra_wrapper(data_module : BaseDataModule , *args, **kwargs):
                buffer = dict()
                if self.param['tra_num_states'] > 1:
                    hist_loss_shape = list(data_module.y.shape)
                    hist_loss_shape[2] = self.param['tra_num_states']
                    buffer['hist_labels'] = data_module.y
                    buffer['hist_preds'] = torch.randn(hist_loss_shape)
                    buffer['hist_loss']  = (buffer['hist_preds'] - buffer['hist_labels'].nan_to_num(0)).square()
                return buffer
            self.setup_wrapper = tra_wrapper
        else:
            self.setup_wrapper = self.none_wrapper
        
    def register_update(self) -> None:
        # first param of wrapper is container, which represent self in ModelData
        if self.key == 'tra':
            def tra_wrapper(data_module : BaseDataModule , *args, **kwargs):
                buffer = dict()
                if self.param['tra_num_states'] > 1:
                    buffer['hist_loss']  = (data_module.buffer['hist_preds'] - 
                                            data_module.buffer['hist_labels'].nan_to_num(0)).square()
                return buffer
            self.update_wrapper = tra_wrapper
        else:
            self.update_wrapper = self.none_wrapper