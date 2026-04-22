from src.res.trading import TrackingPortfolioManager , BacktestPortfolioManager

from src.proj import Logger , Const
from .util import wrap_update

class TradingAPI:
    @classmethod
    def available_ports(cls , backtest : bool | None = None) -> list[str]:
        """
        List tracking and/or backtest portfolio port keys from ``Const.TradingPort``.

        Args:
          backtest: None returns both families; true only backtest; false only tracking.

        Returns:
          Port name strings.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        """
        if backtest is None:
            return list(Const.TradingPort.tracking_ports.keys()) + list(Const.TradingPort.backtest_ports.keys())
        elif backtest:
            return list(Const.TradingPort.backtest_ports.keys())
        else:
            return list(Const.TradingPort.tracking_ports.keys())

    @classmethod
    def backtest_rebuild(cls , port_name : str):
        """
        Rebuild a single backtest portfolio directory and run analysis for *port_name*.

        Args:
          port_name: Key in ``Const.TradingPort.backtest_ports``.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        BacktestPortfolioManager.rebuild(port_name , analyze = True)

    @classmethod
    def backtest_rebuild_all(cls):
        """
        Rebuild every configured backtest portfolio sequentially.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: long
          memory_usage: medium
        """
        for port_name in Const.TradingPort.backtest_ports.keys():
            BacktestPortfolioManager.rebuild(port_name , analyze = True)

    @classmethod
    def update(cls): 
        """
        Refresh tracking and backtest portfolio state for laptop and server deployments.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        def update_trading_ports():
            TrackingPortfolioManager.update()
            BacktestPortfolioManager.update()
        wrap_update(update_trading_ports , 'update trading portfolios')

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs):
        """
        Run portfolio analysis for *port_name* on either backtest or tracking ports.

        Args:
          port_name: Registered port key.
          start: Start date.
          end: End date.
          kwargs: Extra keyword arguments forwarded to managers.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 5
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        if port_name in cls.available_ports(backtest = True):
            return BacktestPortfolioManager.analyze(port_name , start , end , **kwargs)
        elif port_name in cls.available_ports(backtest = False):
            return TrackingPortfolioManager.analyze(port_name , start , end , **kwargs)
        else:
            raise ValueError(f'port name {port_name} is not a valid analyze port')

    @classmethod
    def backtest(cls , port_name_starter : str , start : int | None = None , end : int | None = None , **kwargs): 
        """
        Analyze all backtest ports whose names start with *port_name_starter*.

        Args:
          port_name_starter: Prefix filter against backtest port keys.
          start: Start date.
          end: End date.
          kwargs: Extra keyword arguments forwarded to ``BacktestPortfolioManager.analyze``.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 5
          disable_platforms: []
          execution_time: long
          memory_usage: medium
        """
        available_ports = cls.available_ports(backtest = True)
        ports = [port for port in available_ports if port.startswith(port_name_starter)]
        if ports:
            Logger.stdout(f'multiple backtest ports found for {port_name_starter}: {ports}')
            for port in ports:
                with Logger.Paragraph(f'backtest {port}' , 3):
                    BacktestPortfolioManager.analyze(port , start , end , **kwargs)
        elif len(ports) == 0:
            Logger.error(f'no backtest ports found starting with {port_name_starter}')
            Logger.stdout(f'available backtest ports: {available_ports}')
        
