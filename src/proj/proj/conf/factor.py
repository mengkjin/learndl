# basic variables in factor package
import numpy as np
from typing import Any , Literal

from src.proj.env import MACHINE

__all__ = ['FactorConfig' , 'Factor']

class FactorUpdateConfig:
    @property
    def start(self) -> int:
        """start date of factor update"""
        return 20110101 if MACHINE.updatable else 20241101
    @property
    def end(self) -> int:
        """end date of factor update"""
        return 20401231 if MACHINE.updatable else 20241231
    @property
    def step(self) -> int:
        """step of factor update"""
        return 5
    @property
    def init_date(self) -> int:
        """uniforminit date of factor update"""
        return 20110101
    @property
    def target_dates(self) -> np.ndarray:
        """target dates of factor update"""
        from src.proj import CALENDAR
        return CALENDAR.slice(CALENDAR.td_within(self.init_date , step = self.step) , self.start , self.end)

class RiskModelConfig:
    @property
    def market(self) -> list[str]:
        """market factors"""
        return ['market']
    @property
    def style(self) -> list[str]:
        """style factors"""
        return [
            'size' , 'beta' , 'momentum' , 'residual_volatility' ,
            'non_linear_size' , 'book_to_price' , 'liquidity' , 'earnings_yield' ,
            'growth' , 'leverage'
        ]
    @property
    def indus(self) -> list[str]:
        """industry factors"""
        return [
            'petro' , 'coal' , 'nonferrous' , 'utility' , 'public' , 'steel' , 
            'chemical' , 'construct' , 'cement' , 'material' , 'light' , 'machine' , 
            'power' , 'defense' , 'auto' , 'retail' , 'leisure' , 'appliance' , 
            'textile' , 'health' , 'liqor' , 'food' , 'agro' , 'bank' , 'financial' , 
            'estate' , 'transport' , 'marine' , 'airline' , 'electronic' , 'telecom' , 
            'hardware' , 'software' , 'media' , 'complex'
        ]
    @property
    def common(self) -> list[str]:
        """common factors"""
        return self.market + self.style + self.indus

class BenchmarksConfig:
    @property
    def availables(self) -> list[str]:
        """available benchmarks that can be used"""
        return ['csi300' , 'csi500' , 'csi800' , 'csi1000' , 'csi2000']
    @property
    def defaults(self) -> list[str]:
        """default benchmarks that will be used if not specified"""
        return ['csi500' , 'csi1000']
    @property
    def tests(self) -> list[str]:
        """test benchmarks that will be used if not specified"""
        return ['market' , 'csi300' , 'csi500' , 'csi1000']
    @property
    def categories(self) -> list[str]:
        """categories of benchmarks that can be used"""
        return ['default' , 'none' , 'market' , 'csi300' , 'csi500' , 'csi800' , 'csi1000' , 'csi2000']
    @property
    def none(self) -> list[str]:
        """none benchmarks"""
        return ['none' , 'default' , 'market']

class TradeConfig:
    @property
    def default(self) -> float:
        """default trade cost"""
        return 0.00035
    @property
    def harvest(self) -> float:
        """harvest trade cost"""
        return 0.002
    @property
    def yale(self) -> float:
        """yale trade cost"""
        return 0.00035

class RoundingConfig:
    @property
    def weight(self) -> int:
        """weight rounding"""
        return 6
    @property
    def exposure(self) -> int:
        """exposure rounding"""
        return 6
    @property
    def contribution(self) -> int:
        """contribution rounding"""
        return 8
    @property
    def ret(self) -> int:
        """ret rounding"""
        return 8
    @property
    def turnover(self) -> int:
        """turnover rounding"""
        return 8

class PortfolioOptimizationConfig:
    @property
    def default(self) -> dict[str , Any]:
        """default portfolio optimization config"""
        return MACHINE.configs('util' , 'factor' , 'default_opt_config')
    @property
    def custom(self) -> dict[str , Any]:
        """custom portfolio optimization config"""
        return MACHINE.configs('util' , 'factor' , 'custom_opt_config')

class _StockFactorDefinitionMetaType:
    def __get__(self,instance,owner) -> list[str]:
        return [
            'stock' , 'market' , 'affiliate' , 'pooling'
        ]

    def __set__(self,instance,value):
        raise AttributeError(f'{instance.__class__.__name__} is read-only attributes')

class _StockFactorDefinitionCat0:
    def __get__(self,instance,owner) -> list[str]:
        return [
            'fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative' ,
            'market' ,
            'risk' , 'external' ,
            'pooling' , 
        ]

    def __set__(self,instance,value):
        raise AttributeError(f'{instance.__class__.__name__} is read-only attributes')

class _StockFactorDefinitionCat1:
    def __get__(self,instance,owner) -> dict[str , list[str] | None]:
        return {
            'external' : ['sellside'] ,
            'risk' : ['style'] ,
            'pooling' : ['weighted' , 'nonlinear'] ,
            'fundamental' : ['quality' , 'growth' , 'value' , 'earning'] ,
            'analyst' : ['surprise' , 'coverage' , 'forecast' , 'adjustment'] ,
            'high_frequency' : ['hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity'] ,
            'behavior' : ['momentum' , 'volatility' , 'correlation' , 'liquidity'] ,
            'money_flow' : ['holding' , 'trading'] ,
            'alternative' : None ,
            'market' : ['market_event']
        }
    def __set__(self,instance,value):
        raise AttributeError(f'{instance.__class__.__name__} is read-only attributes')

class CategoryError(Exception): 
    ...

class StockFactorDefinitionConfig:
    _META = _StockFactorDefinitionMetaType()
    _CAT0 = _StockFactorDefinitionCat0()
    _CAT1 = _StockFactorDefinitionCat1()

    @property
    def meta_type(self) -> list[str]:
        """meta type of stock factor"""
        return self._META
    
    @property
    def category0(self) -> list[str]:
        """category0 of stock factor"""
        return self._CAT0
    @property
    def category1(self) -> dict[str , list[str] | None]:
        """category1 of stock factor"""
        return self._CAT1

    @classmethod
    def cat0_to_meta(cls , category0 : str) -> Literal['stock' , 'market' , 'affiliate' , 'pooling']:
        if category0 not in cls._CAT0:
            raise CategoryError(f'category0 is should be in {cls._CAT0}, but got {category0}')
        if category0 == 'market':
            return 'market'
        elif category0 in ['risk' , 'external']:
            return 'affiliate'
        elif category0 == 'pooling':
            return 'pooling'
        else:
            return 'stock'
            
    @classmethod
    def cat0_to_cat1(cls , category0 : str) -> list[str] | None:
        """Get the possible category1 of the category0 of stock factor"""
        return cls._CAT1[category0]

    @classmethod
    def cat1_to_cat0(cls , category1 : str) -> str:
        """Get the category0 given category1 of stock factor"""
        match category1:
            case 'quality' | 'growth' | 'value' | 'earning':
                category0 = 'fundamental'
            case 'surprise' | 'coverage' | 'forecast' | 'adjustment':
                category0 = 'analyst'
            case 'hf_momentum' | 'hf_volatility' | 'hf_correlation' | 'hf_liquidity':
                category0 = 'high_frequency'
            case 'momentum' | 'volatility' | 'correlation' | 'liquidity':
                category0 = 'behavior'
            case 'holding' | 'trading':
                category0 = 'money_flow'
            case 'market_event':
                category0 = 'market'
            case 'style':
                category0 = 'risk'
            case 'sellside':
                category0 = 'external'
            case 'weighted' | 'nonlinear':
                category0 = 'pooling'
            case _:
                raise ValueError(f'undefined category1: {category1}')
        return category0

    @classmethod
    def validate_categories(cls , category0 : str , category1 : str) -> None:
        if category0 not in cls._CAT0:
            raise CategoryError(f'category0 is should be in {cls._CAT0}, but got {category0}')

        if not category1:
            raise CategoryError('category1 is not set')

        if (category1_list := cls._CAT1[category0]):
            if category1 not in category1_list:
                raise CategoryError(f'category1 is should be in {category1_list}, but got {category1}')


class FactorConfig:
    def __init__(self):
        self._update = FactorUpdateConfig()
        self._risk = RiskModelConfig()
        self._bench = BenchmarksConfig()
        self._trade = TradeConfig()
        self._rounding = RoundingConfig()
        self._optim = PortfolioOptimizationConfig()
        self._stock = StockFactorDefinitionConfig()

    @property
    def UPDATE(self):
        """config of factor update , include start , end , step , init_date"""
        return self._update

    @property
    def RISK(self):
        """config of risk model , include market , style , indus , common"""
        return self._risk

    @property
    def BENCH(self):
        """config of benchmarks , include availables , defaults , categories , none"""
        return self._bench

    @property
    def TRADE(self):
        """config of trade cost , include default , harvest , yale"""
        return self._trade

    @property
    def ROUNDING(self):
        """config of rounding , include weight , exposure , contribution , ret , turnover"""
        return self._rounding

    @property
    def OPTIM(self):
        """config of portfolio optimization , include default , custom"""
        return self._optim

    @property
    def STOCK(self):
        """config of stock factor definition , include category0 , category1 , cat0_to_cat1 , cat1_to_cat0 , validate_categories"""
        return self._stock
    

Factor = FactorConfig()