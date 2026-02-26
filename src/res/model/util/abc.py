import re
from typing import Any , Literal
from pathlib import Path

from src.proj import PATH , MACHINE , Logger
from src.res.factor.calculator.factor_calc import FactorCalculator
from src.res.algo import AlgoModule


TYPE_MODULE_TYPES = Literal['nn' , 'boost' , 'factor' ,'']
MODULE_TYPES : list[TYPE_MODULE_TYPES] = ['nn' , 'boost' , 'factor' , '']
MODEL_SETTINGS = MACHINE.configs('setting' , 'model')

def parse_model_input(model_input : Path | str | None) -> dict[str,Any]:
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

def combine_full_name(st : str , module_type : str , model_module : str , model_clean_name : str , model_name_index : str | int) -> str:
    """
    combine a full model name into (st , module_type , model_module , model_clean_name , model_name_index)
    """
    if is_null_module_type(module_type):
        assert model_module == model_clean_name , f'model_module {model_module} and model_clean_name {model_clean_name} are not the same a {module_type} model'
        assert model_name_index == '' or (isinstance(model_name_index , int) and model_name_index == 0), f'model_name_index {model_name_index} is not empty as a {module_type} model'
        name = f'{module_type}@{model_module}'
    else:
        name = f'{module_type}@{model_module}@{model_clean_name}'
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
        model_module = text.split('@')[0]
        module_type = model_module_type(model_module)
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
            model_module , model_clean_name = remain.split('@' , 1)
        else:
            assert is_null_module_type(module_type) , f'Invalid module type [{module_type}] when model_module/model_clean_name is {remain}'
            model_module = remain
            model_clean_name = remain
        if is_null_module_type(module_type):
            assert model_module == model_clean_name , f'model_module {model_module} and model_clean_name {model_clean_name} are not the same for {text} as a {module_type} model'
            assert model_name_index == '' , f'model_name_index {model_name_index} is not empty for {text} as a {module_type} model'
    except ValueError as e:
        Logger.error(f'Failed to split full name [{text}]')
        raise e
    return st , module_type , model_module , model_clean_name , model_name_index

def model_module_type(model_module : str) -> TYPE_MODULE_TYPES:
    f"""
    detect module type from various forms of model_module if it is a valid nn / boost / factor / db type
    if nn / nn@nn_module_name / nn_module_name , it is a valid nn type
    if boost / boost@boost_module_name / boost_module_name , it is a valid boost type
    if factor / factor@factor_module_name / factor_module_name , it is a valid factor type
    if db / db@db_module_name / db_module_name , it is a valid db type
    if model_module is not a valid type, return ''
    """
    if model_module in MODULE_TYPES:
        return model_module
    elif '@' in model_module:
        match = re.match(r'^(nn|boost|db|factor)\@(.*)$', model_module)
        assert match , f'Invalid module type for [{model_module}]'
        module_type , model_module = match.group(1) , match.group(2)
        assert module_type in MODULE_TYPES , f'Invalid module type [{module_type}]'
        return module_type
    else:
        module_type = AlgoModule.module_type(model_module , raise_error = False)
        if module_type == '':
            module_type = check_null_module_type(model_module)
        return module_type

def check_null_module_type(model_module : str) -> TYPE_MODULE_TYPES:
    """detect module type from model_module if it is not a valid nn / boost type. If one is both, return factor first"""
    if model_module == 'factor' or model_module.startswith('factor@') or model_module in FactorCalculator.all_factors():
        return 'factor'
    else:
        return ''

def is_null_module_type(module_type : str) -> bool:
    """check if model_module is a valid null module type"""
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