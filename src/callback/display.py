from ..util.classes import BaseCallBack
from ..util import LoaderWrapper

class LoaderDisplay(BaseCallBack):
    def __init__(self , batch_interval = 1) -> None:
        super().__init__()
        self._interval = batch_interval
    @staticmethod
    def _loader(Mmod) -> LoaderWrapper: return Mmod.dataloader
    def on_train_epoch_start(self , Mmod):      self._loader(Mmod).init_tqdm('Train Ep#{ep:3d} loss : {ls:.5f}')
    def on_validation_epoch_start(self , Mmod): self._loader(Mmod).init_tqdm('Valid Ep#{ep:3d} score : {sc:.5f}')
    def on_test_model_type_start(self , Mmod):  self._loader(Mmod).init_tqdm('Test {mt} {dt} score : {sc:.5f}')
    def on_train_batch_end(self , Mmod):
        if Mmod.pipe.epoch % self._interval == 0:
            self._loader(Mmod).display(ep=Mmod.pipe.epoch, ls=Mmod.pipe.aggloss)
    def on_validation_batch_end(self , Mmod):
        if Mmod.pipe.epoch % self._interval == 0:
            self._loader(Mmod).display(ep=Mmod.pipe.epoch, sc=Mmod.pipe.aggscore)
    def on_test_batch_end(self , Mmod):
        if Mmod.pipe.epoch % self._interval == 0:
            self._loader(Mmod).display(dt=Mmod.test_dates[Mmod.batch_idx] , mt=Mmod.model_type , sc=Mmod.pipe.aggscore)