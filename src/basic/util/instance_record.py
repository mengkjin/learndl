from typing import Any

class InstanceRecord:
    '''singleton class to record instances'''
    _instance = None
    _slots = ['trainer' , 'account']

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.record : dict[str, Any] = {}
        print(f'src.basic.INSTANCE_RECORD can be accessed to check {self._slots}')

    def __getitem__(self , key: str , default: Any = None) -> Any:
        return self.record.get(key , default)

    def __setitem__(self , key: str , value: Any) -> None:
        assert key in self._slots , f'key {key} is not in {self._slots}'
        self.record[key] = value

    def __contains__(self , key: str) -> bool:
        return key in self.record
    
    def __repr__(self):
        return f'{self.__class__.__name__}(names={list(self.record.keys())})'
    
    @property
    def trainer(self):
        return self['trainer']
    
    @property
    def account(self):
        return self['account']

INSTANCE_RECORD = InstanceRecord()
