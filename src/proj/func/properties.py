from typing import Any

def _get_obj_attr(obj : Any , attr : str) -> Any:
    if hasattr(obj , attr):
        obj_attr = getattr(obj , attr)
        if callable(obj_attr):
            obj_attr = obj_attr()
        return obj_attr
    else:
        return None

def empty(*objs : Any) -> bool:
    obj_empty : list[bool] = []
    for obj in objs:
        if obj is None:
            obj_empty.append(True)
        elif hasattr(obj , 'empty'):
            obj_empty.append(bool(_get_obj_attr(obj , 'empty')))
        elif hasattr(obj , 'size'):
            obj_empty.append(_get_obj_attr(obj , 'size') == 0)
        elif hasattr(obj , '__len__'):
            obj_empty.append(_get_obj_attr(obj , '__len__') == 0)
        elif hasattr(obj , '__bool__'):
            obj_empty.append(not bool(_get_obj_attr(obj , '__bool__')))
        else:
            raise ValueError(f'No empty method for {type(obj)}')
    return all(obj_empty)

def shape(obj : Any , keys : list[str] | None = None) -> Any:
    if obj is None: 
        return []
    elif keys is not None:
        return {key:shape(getattr(obj , key)) for key in keys}
    elif hasattr(obj , 'shape'): 
        obj_shape = getattr(obj , 'shape')
        if callable(obj_shape):
            obj_shape = obj_shape()
        return obj_shape
    elif isinstance(obj , (list , tuple)): 
        return tuple([shape(x) for x in obj])
    elif isinstance(obj , dict):
        return {key:shape(value) for key,value in obj.items()}
    else: 
        return [f'No shape({type(obj)})']

def max_date(date : Any) -> int: 
    if date is None or len(date) == 0:
        return -1 
    else:
        return int(max(date))

def min_date(date : Any) -> int: 
    if date is None or len(date) == 0:
        return 99991231
    else:
        return int(min(date))