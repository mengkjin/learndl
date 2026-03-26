from src.proj import Logger
class Test:
    def __init__(self):
        Logger.only_once('test' , object = self.__class__ , mark = 'test' , printer = Logger.success)
    def __repr__(self):
        return 'Test'
