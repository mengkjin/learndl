"""
Direct calls related to testing if code is working as expected.
"""

from src.api.util.direct_call import DirectCall
from src.proj.util.functional.ask import AskFor

class TestCode(DirectCall):
    """Running tests for the project code."""
    category = 'Test'
    @classmethod
    def _test_logger(cls):
        """Test the logger."""
        from src.proj import Logger
        Logger.test_logger()
    @classmethod
    def _test_quick_train(cls):
        """Test quick training of a model."""
        from src.api.pkgs.model import ModelAPI
        ModelAPI.train_model(short_test = True)
    @classmethod
    def _test_parallel_factor_calculation(cls):
        """Test parallel factor calculation."""
        from src.api.calls.test.parallel_factor_calculation import test_parallel_factor_calculation
        test_parallel_factor_calculation()
    @classmethod
    def _menu_for_operations(cls):
        testors = [getattr(cls, name) for name in dir(cls) if name.startswith('_test_')]
        options = [testors.__doc__ for testors in testors]
        flag = AskFor.Options(options , confirm = False , multiple = False , title = f'What project tests to conduct?')
        if flag.result is None:
            return flag
        selection = options.index(flag.result)
        flag = testors[selection]()
        return flag

    def run(self) -> None:
        for loop in AskFor.LoopTillExit(False):
            loop.set_flag(self._menu_for_operations())
