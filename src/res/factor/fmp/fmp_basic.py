import os
from typing import Any , Literal

from ..util import Portfolio , Benchmark , AlphaModel , Port

from .generator import TopStocksPortfolioCreator , ScreeningPortfolioCreator

def parse_full_name(full_name : str):
    components = full_name.split('.')
    assert len(components) >= 5 , f'Full name must have at least 4 components: {full_name}'
    prefix = components[0]
    category = prefix.lower()
    assert category in ['optim' , 'top' , 'screen'] , f'Unknown category: {category}'
    factor_name , benchmark , strategy = components[1:4]
    suffix = '.'.join(components[4:])
    lag = int(components[4].split('lag')[-1])
    elements = {
        'category'    : category ,
        'prefix'      : prefix ,
        'factor_name' : factor_name ,
        'benchmark'   : benchmark ,
        'strategy'    : strategy ,
        'suffix'      : suffix ,
        'lag'         : lag ,
    }
    elements['suffixes'] = components[5:]
    if category == 'top' and strategy.startswith('Top'):
        elements['n_best'] = int(strategy.split('Top')[-1].replace('_',''))
    return elements

def get_prefix(category : Literal['optim' , 'top' , 'screen']): return category.title()
    
def get_factor_name(alpha : AlphaModel | str):
    return alpha.name if isinstance(alpha , AlphaModel) else alpha

def get_benchmark(benchmark : Portfolio | Benchmark | Port | str | None = None): 
    if benchmark is None:
        benchmark = Portfolio()
    elif isinstance(benchmark , str):
        benchmark = Benchmark(benchmark)
    elif isinstance(benchmark , Port):
        benchmark = Portfolio.from_ports(benchmark)
    return benchmark

def get_benchmark_name(benchmark : Portfolio | Benchmark | str | None):
    if benchmark is None:
        return 'default'
    elif isinstance(benchmark , (Portfolio , Benchmark)):
        return benchmark.name
    elif isinstance(benchmark , str):
        return benchmark
    else:
        raise ValueError(f'Unknown benchmark type: {type(benchmark)}')

def get_strategy_name(category : Literal['optim' , 'top' , 'screen'] , strategy : str = 'default' , kwargs : dict[str,Any] | None = None):
    kwargs = kwargs or {}
    if not strategy or strategy == 'default':
        if category == 'top':
            n = kwargs.get('n_best' , TopStocksPortfolioCreator.DEFAULT_N_BEST)
            strategy = f'Top{n:_>3d}'
        elif category == 'screen':
            ratio = kwargs.get('screen_ratio' , ScreeningPortfolioCreator.DEFAULT_SCREEN_RATIO)
            strategy = f'Screen{ratio * 100:.0f}%'
        elif category == 'optim':
            strategy = os.path.basename(kwargs['config_path']) if 'config_path' in kwargs else 'default'
        else:
            raise ValueError(f'Unknown category: {category}')
    assert '.' not in strategy , f'To avoid conflict with factor name, strategy name cannot contain dot: {strategy}'
    return strategy

def get_suffix(lag : int , suffixes : list[str] | str | None = None): 
    suffixes = suffixes or []
    if isinstance(suffixes , str): 
        suffixes = [suffixes]
    return '.'.join([f'lag{lag}' , *suffixes])

def get_full_name(category : Literal['optim' , 'top' , 'screen'] , alpha : AlphaModel | str , 
                  benchmark : Portfolio | Benchmark | str | None = None , 
                  strategy : str = 'default' , suffixes : list[str] | str | None = None , lag : int = 0 , **kwargs):
    suffixes = suffixes or []
    return '.'.join([
        get_prefix(category) , 
        get_factor_name(alpha) , 
        get_benchmark_name(benchmark) , 
        get_strategy_name(category , strategy , kwargs) , 
        get_suffix(lag , suffixes)
    ])


def get_port_index(full_name : str):
    elements = parse_full_name(full_name)

    default_index : dict[str,Any] = {
        'prefix'      : elements['prefix'] ,
        'factor_name' : elements['factor_name'] ,
        'benchmark'   : elements['benchmark'] ,
        'strategy'    : elements['strategy'] ,
        'suffix'      : elements['suffix'] ,
    }
    default_index['lag'] = elements['lag']
    if 'n_best' in elements:
        default_index['topN'] = elements['n_best']
    return default_index