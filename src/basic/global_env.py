class _Silence:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self): self.silent = False
    def __bool__(self): return bool(GLOB_ENV['silent'])
    def __enter__(self) -> None: 
        self.raw_silent = self.silent
        self.silent = True
    def __exit__(self , *args) -> None: 
        self.silent = self.raw_silent

GLOB_ENV = {
    'tushare_indus_encoding' : 'gbk' ,
    'silent' : _Silence() ,
}