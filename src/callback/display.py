from ..util.classes import BaseCallBack
from ..util import LoaderWrapper

class LoaderDisplay(BaseCallBack):
    def __init__(self , model_module , batch_interval = 1) -> None:
        super().__init__(model_module)
        self._interval = batch_interval
        print(f'{self._infomation()}({batch_interval}) , this is a little bit slow!')
    def _dl(self) -> LoaderWrapper: return self.model_module.dataloader
    def on_train_epoch_start(self):      self._dl().init_tqdm('Train Ep#{ep:3d} loss : {ls:.5f}')
    def on_validation_epoch_start(self): self._dl().init_tqdm('Valid Ep#{ep:3d} score : {sc:.5f}')
    def on_test_model_type_start(self):  self._dl().init_tqdm('Test {mt} {dt} score : {sc:.5f}')
    def on_train_batch_end(self):
        if self.model_module.pipe.epoch % self._interval == 0:
            self._dl().display(ep=self.model_module.pipe.epoch, ls=self.model_module.pipe.aggloss)
    def on_validation_batch_end(self):
        if self.model_module.pipe.epoch % self._interval == 0:
            self._dl().display(ep=self.model_module.pipe.epoch, sc=self.model_module.pipe.aggscore)
    def on_test_batch_end(self):
        if self.model_module.pipe.epoch % self._interval == 0:
            self._dl().display(dt=self.model_module.test_dates[self.model_module.batch_idx] , 
                               mt=self.model_module.model_type , sc=self.model_module.pipe.aggscore)