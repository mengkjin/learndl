from typing import Any

class classproperty:
    '''a class property decorator'''
    def __init__(self,method):
        self.method = method
 
    def __get__(self,instance,owner) -> Any:
        return self.method(owner)
    
class classproperty_str:
    '''a class property decorator that returns a string , useful for some pylance type hint'''
    def __init__(self,method):
        self.method = method
 
    def __get__(self,instance,owner) -> str:
        return str(self.method(owner))
