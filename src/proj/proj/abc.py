from typing import Any
from src.proj.env import MACHINE
_project_settings = MACHINE.configs('setting' , 'project')
class ProjectSetting:
    def __init__(self , key : str , default : Any = None):
        self.key = key
        self.default = default
    def __get__(self , instance, owner) -> bool:
        return self.get(self.key , self.default)
    @staticmethod
    def get(key : str , default : Any = None) -> Any:
        return _project_settings.get(key , default)
