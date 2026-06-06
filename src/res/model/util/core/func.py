from __future__ import annotations
import re
from typing import Any , Literal
from pathlib import Path

from src.proj import PATH , Base
from src.res.factor.calculator.factor_calc import FactorCalculator
from src.res.algo import AlgoModule

__all__ = [
    'TYPE_MODULE_TYPES' , 'MODULE_TYPES' , 'epoch_key' , 'attempt_key' ,
    'parse_model_input' , 'combine_full_name' , 'split_full_name' , 
    'model_module_type' , 'check_null_module_type' , 'is_null_module_type' , 'search_existing_models']

TYPE_MODULE_TYPES = Literal['nn' , 'boost' , 'factor' ,'']
MODULE_TYPES : list[TYPE_MODULE_TYPES] = ['nn' , 'boost' , 'factor' , '']

def epoch_key(epoch : int , phase : int = 0) -> str:
    return f'Ph{phase} Ep{epoch}'

def attempt_key(attempt : int , redo : int = 0) -> str:
    return f'Trial{attempt}-{redo}'

def parse_model_input(model_input : Base.types.strPath | None) -> dict[str,Any]:
    """return full model name and root path of a given model input"""
    if model_input is None or model_input == '' or model_input == Path(''):
        return {
            'full_name' : '' , 
            'st' : '' ,
            'module_type' : '' ,
            'module_name' : '' ,
            'model_clean_name' : '' ,
            'model_name_index' : ''
        }
    elif isinstance(model_input , Path):
        assert model_input.is_relative_to(PATH.model), f'model_input {model_input} is not a subdirectory of {PATH.model}'
        if model_input.parent == PATH.model_st:
            return parse_model_input(f'st@{model_input.name}')
        else:
            return parse_model_input(f'{model_input.parent.name}@{model_input.name}')
    elif isinstance(model_input , str):
        if model_input.startswith('models/'):
            model_input = model_input.removeprefix('models/')
        st , module_type , module_name , model_clean_name , index = split_full_name(model_input.replace('/' , '@').replace('\\' , '@'))
        full_name = combine_full_name(st , module_type , module_name , model_clean_name , index)
        return {
            'full_name' : full_name ,
            'st' : st ,
            'module_type' : module_type ,
            'module_name' : module_name ,
            'model_clean_name' : model_clean_name ,
            'model_name_index' : index
        }
    else:
        raise ValueError(f'Invalid model input [{model_input}]')

def combine_full_name(st : str , module_type : str , module_name : str , model_clean_name : str , model_name_index : str | int) -> str:
    """
    combine a full model name into (st , module_type , module_name , model_clean_name , model_name_index)
    """
    if is_null_module_type(module_type):
        assert module_name == model_clean_name , f'module_name {module_name} and model_clean_name {model_clean_name} are not the same a {module_type} model'
        assert model_name_index == '' or (isinstance(model_name_index , int) and model_name_index == 1), f'model_name_index {model_name_index} is not empty as a {module_type} model'
        name = f'{module_type}@{module_name}'
    else:
        name = f'{module_type}@{module_name}@{model_clean_name}'
    if (isinstance(model_name_index , str) and model_name_index) or (isinstance(model_name_index , int) and model_name_index >= 2):
        name = f'{name}@{model_name_index}'
    if st:
        name = f'{st}@{name}'
    return name.replace('/' , '@').replace('\\' , '@')

def split_digits_suffix(text : str) -> tuple[str,str]:
    """
    Splits off a trailing '.digits' suffix using regex.
    Returns (before, number) if found, else (text, '').
    """
    match = re.search(r'\@(\d+)$', text)
    if match:
        return text[:match.start()], match.group(1)
    return text, ''

def split_st_prefix(text : str) -> tuple[str,str]:
    """
    If the string starts with 'aa.' or 'bb.', returns (prefix, rest).
    Otherwise, returns ('', text).
    """
    match = re.match(r'^(st)\@(.*)$', text)
    if match:
        return match.group(1), match.group(2)
    return '', text

def split_module_type_prefix(text : str) -> tuple[TYPE_MODULE_TYPES,str]:
    """
    If the string starts with 'aa.' or 'bb.', returns (prefix, rest).
    Otherwise, returns ('', text).
    """
    match = re.match(r'^(nn|boost|db|factor)\@(.*)$', text)
    if match:
        module_type : Any = match.group(1)
        remain : str = match.group(2)
        assert module_type in ['nn' , 'boost' , 'factor'] , f'Invalid module type [{module_type}]'
    else:
        module_name = text.split('@')[0]
        module_type = model_module_type(module_name)
        if module_type == '':
            module_type , remain = search_existing_models(text)
        else:
            remain = text
    return module_type, remain

def split_full_name(text : str) -> tuple[str,TYPE_MODULE_TYPES,str,str,str]:
    """
    split a full model name into (st , module_type , module_name , model_name , index)
    """
    if not text:
        return '' , '' , '' , '' , ''
    try:
        st , remain = split_st_prefix(text)
        module_type , remain = split_module_type_prefix(remain)
        remain , model_name_index = split_digits_suffix(remain)
        if '@' in remain:
            module_name , model_clean_name = remain.split('@' , 1)
        else:
            assert is_null_module_type(module_type) , f'Invalid module type [{module_type}] when module_name/model_clean_name is {remain}'
            module_name = remain
            model_clean_name = remain
        if is_null_module_type(module_type):
            assert module_name == model_clean_name , f'module_name {module_name} and model_clean_name {model_clean_name} are not the same for {text} as a {module_type} model'
            assert model_name_index == '' , f'model_name_index {model_name_index} is not empty for {text} as a {module_type} model'
    except ValueError as e:
        raise ValueError(f'Failed to split full name [{text}] : {e}')
    return st , module_type , module_name , model_clean_name , model_name_index

def model_module_type(module_name : str) -> TYPE_MODULE_TYPES:
    f"""
    detect module type from various forms of module_name if it is a valid nn / boost / factor / db type
    if nn / nn@nn_module_name / nn_module_name , it is a valid nn type
    if boost / boost@boost_module_name / boost_module_name , it is a valid boost type
    if factor / factor@factor_module_name / factor_module_name , it is a valid factor type
    if db / db@db_module_name / db_module_name , it is a valid db type
    if module_name is not a valid type, return ''
    """
    if module_name in MODULE_TYPES:
        return module_name
    elif '@' in module_name:
        match = re.match(r'^(nn|boost|db|factor)\@(.*)$', module_name)
        assert match , f'Invalid module type for [{module_name}]'
        module_type , module_name = match.group(1) , match.group(2)
        assert module_type in MODULE_TYPES , f'Invalid module type [{module_type}]'
        return module_type
    else:
        module_type = AlgoModule.module_type(module_name , raise_error = False)
        if module_type == '':
            module_type = check_null_module_type(module_name)
        return module_type

def check_null_module_type(module_name : str) -> TYPE_MODULE_TYPES:
    """detect module type from module_name if it is not a valid nn / boost type. If one is both, return factor first"""
    if module_name == 'factor' or module_name.startswith('factor@') or module_name in FactorCalculator.all_factors():
        return 'factor'
    else:
        return ''

def is_null_module_type(module_type : str) -> bool:
    """check if module_name is a valid null module type"""
    return module_type in ['factor']

def search_existing_models(model_name : str) -> tuple[TYPE_MODULE_TYPES,str]:
    """search existing models in nn and boost directories to detect module type and model name"""
    for model in PATH.model_nn.glob(f'*{model_name}'):
        if model.is_dir() and not model.name.startswith('.'):
            return 'nn' , model.name
    for model in PATH.model_boost.glob(f'*{model_name}'):
        if model.is_dir() and not model.name.startswith('.'):
            return 'boost' , model.name
    raise ValueError(f'Cannot find existing model for [{model_name}]')