import numpy as np
from dataclasses import dataclass
from typing import Literal
from src.data import DATAVENDOR

@dataclass
class TradeSuggestion:
    secid : int
    secname : str
    direction : Literal['buy' , 'sell']
    price : float
    pct : float

    def format(self):
        return f' >> Suggest {self.direction.title():>4s} {self.secid:06d} [{self.secname:^4s}] at {self.price:04.2f} ({self.pct:+02.2%})'

    @classmethod
    def generate(cls , secid : np.ndarray | list[int] , d : int , direction : Literal['buy' , 'sell']):
        secid = np.array(secid) if isinstance(secid , list) else secid
        secname = DATAVENDOR.INFO.secname(secid).tolist()
        if_st = np.isin(secid , DATAVENDOR.INFO.get_st(d)['secid'].to_numpy())
        if_registered = (secid >= 688000) + ((secid >= 300000) * (secid < 400000)) == 1

        pcts = np.full(len(secid) , 0.1)
        pcts[if_registered] = 0.2
        pcts[if_st] = 0.05
        

        cp = DATAVENDOR.get_cp(secid , d)
        if direction == 'buy':
            price = cp * (1 + pcts).round(2)
        else:
            price = cp * (1 - pcts).round(2)
        
        pcts = pcts / cp - 1
        suggestions = [cls(secid[i] , secname[i] , direction , price[i] , pcts[i]) for i in range(len(secid))]
        return suggestions