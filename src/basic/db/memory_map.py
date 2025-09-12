import pandas as pd
import numpy as np
import shutil
from pathlib import Path

def get_shape(obj) -> tuple[int, ...]:
    if hasattr(obj , 'shape'):
        return obj.shape
    return (len(obj),)


def save_min_to_mmap(
        df : pd.DataFrame , path : str | Path , 
        overwrite = False ,
        key_names : tuple[str , str] = ('secid' , 'minute') ,
        key_dtypes = ('int32' , 'int32')):
    path = Path(path) if isinstance(path , str) else path
    if path.suffix in ['.feather' , '.csv' , '.parquet']: 
        path = path.with_suffix('')
    if not overwrite and path.exists():
        raise FileExistsError(f'file already exists: {path}')
    else:
        if path.exists():
            shutil.rmtree(path)
        else:
            path.mkdir(parents=True , exist_ok=True)
    if df.columns.nlevels == 1:
        df = df.pivot_table(index = key_names[0] , columns = key_names[1])
    key1 = df.index.values
    assert isinstance(df.columns , pd.MultiIndex) , f'columns must be a MultiIndex: {df}'
    key2 = df.columns.levels[1]
    features = df.columns.levels[0]
    
    mmap = np.memmap(path.joinpath(f'{key_names[0]}.mmap') , dtype=key_dtypes[0], mode='w+', shape=get_shape(key1))
    mmap[:] = key1

    mmap = np.memmap(path.joinpath(f'{key_names[1]}.mmap') , dtype=key_dtypes[1], mode='w+', shape=get_shape(key2))
    mmap[:] = key2

    for feat in features:
        data = df[feat].values
        mmap = np.memmap(path.joinpath(f'{feat}.mmap'), dtype='float32', mode='w+', shape=data.shape)
        mmap[:] = data

def load_min_from_mmap(path : str | Path , features : list | None = None , 
                       key_names : tuple[str , str] = ('secid' , 'minute') ,
                       key_dtypes = ('int32' , 'int32')):
    path = Path(path) if isinstance(path , str) else path
    key1 = np.memmap(path.joinpath(f'{key_names[0]}.mmap'), dtype=key_dtypes[0], mode='r')
    key2 = np.memmap(path.joinpath(f'{key_names[1]}.mmap'), dtype=key_dtypes[1], mode='r')
    
    # 读取特征数据
    if features is None:
        features = [f.name.removesuffix('.mmap') for f in path.iterdir() 
                    if f.name.endswith('.mmap') and f.name.removesuffix('.mmap') not in key_names]
    
    shape = (len(key1) , len(key2))
    data = {feat:np.memmap(path.joinpath(f'{feat}.mmap'), dtype='float32', mode='r').reshape(*shape) for feat in features}
    df = pd.concat([pd.DataFrame(data[feature] , index = key1 , columns = key2) for feature in features], axis=1 , keys = features)
    return df