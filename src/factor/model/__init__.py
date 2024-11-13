from .cne5 import TuShareCNE5_Calculator

class FactorModelUpdater:
    @classmethod
    def proceed(cls):
        task_cne5 = TuShareCNE5_Calculator()
        task_cne5.Update('exposure')
        task_cne5.Update('risk')
