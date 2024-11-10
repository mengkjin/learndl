class Silence:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self): 
        self._silent : bool = False
        self._enable : bool = True
    @property
    def silent(self): 
        return self._silent and self._enable
    def __bool__(self): 
        return self.silent
    def __enter__(self) -> None: 
        self._raw_silent = self._silent
        self._silent = True
    def __exit__(self , *args) -> None: 
        self._silent = self._raw_silent
    def disable(self): 
        self._enable = False
    def enable(self): 
        self._enable = True

SILENT = Silence()