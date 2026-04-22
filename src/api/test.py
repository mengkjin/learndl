from src.proj import Logger
from typing import Literal , Any

class TestAPI:
    @classmethod
    def test(
        cls , names : str = '' , factor_type : Literal['factor' , 'pred'] = 'factor' , 
        benchmark : list[str] | str | Literal['defaults'] = 'defaults' , 
        start : int = 20240101 , end : int | None = None , step : int = 5 ,
        write_down : bool = True , display_figs : bool = False , indent : int = 0 , vb_level : Any = 1 , 
    ):
        """
        Run top fmp test on factors resolved via ``get_factor``.

        Args:
          names : Factor name(s) to test. If '' or 'random', a random factor will be used; Factor name(s) will be split by ','.
          factor_type : Factor type ('factor' or 'pred').
          benchmark : Benchmark name(s) to test. If 'defaults', will use all default benchmarks. If a string, will be split by ','.
          start : Start date for the test.
          end : End date for the test.
          step : Step size for the test.
          write_down : Whether to write down the test results.
          display_figs : Whether to display the test figures.
          indent : Indent level for the test.
          vb_level : Verbosity level for the test.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          override_arg_attr:
            benchmark :
              type: str
              default: defaults
          risk: read_only
          lock_num: 2
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        """
        with Logger.Paragraph('test top fmp' , 3):
            string = f'test {names} with {factor_type} and {benchmark} from {start} to {end} with step {step} and write_down {write_down} and display_figs {display_figs} and indent {indent} and vb_level {vb_level}'
            Logger.info(string)
        return string