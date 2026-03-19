from src.proj import CALENDAR , Logger , Proj
from .trading_port import BacktestPort

class TradingPortfolioBacktestor:
    @classmethod
    def available_ports(cls) -> list[str]:
        return list(BacktestPort.candidate_ports.keys())

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = BacktestPort.load(port_name)
        date = end if end is not None else CALENDAR.updated()
        return tp.build(date).analyze(start = start , end = end , **kwargs)

    @classmethod
    def rebuild(cls , port_name : str , date : int | None = None , export = True , indent : int = 1 , vb_level : int = 2):
        tp = BacktestPort.load(port_name)
        date = date if date is not None else CALENDAR.updated()
        tp.rebuild(date , export = export , indent = indent , vb_level = vb_level)
        tp.analyze(end = date)
        return tp

    @classmethod
    def update(cls , indent : int = 0 , vb_level : int = 1):
        Logger.note(f'Update: {cls.__name__} since last update!' , indent = indent)
        date = CALENDAR.updated()
            
        updated_ports = {name:BacktestPort.load(name).build(date , indent = indent + 1 , vb_level = Proj.vb.max) 
                         for name in BacktestPort.candidate_ports}
        updated_ports = {name:tp for name,tp in updated_ports.items() if not tp.new_ports[date].empty}
            
        if len(updated_ports) == 0: 
            Logger.alert1(f'No backtest portfolios updated on {date}' , indent = indent + 1)
        else:
            Logger.success(f'{len(updated_ports)} Backtest portfolios updated on {date}' , indent = indent + 1 , vb_level = vb_level)
        for port_name in updated_ports:
            updated_ports[port_name].analyze(key_fig = '' , vb_level = Proj.vb.max)