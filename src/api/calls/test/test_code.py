"""
Direct calls related to testing if code is working as expected.
"""

from src.api.util.direct_call import DirectCall
from src.proj.util.cli import AskFor

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
        options = [testor.__doc__ for testor in testors]
        option_help = {
            doc: doc for doc in options if doc
        }
        flag = AskFor.Options(
            options , confirm = False , multiple = False ,
            title = f'What project tests to conduct?',
            help_description='Smoke tests for logging, quick training, and parallel factor calculation.',
            option_help=option_help,
        )
        if flag.result is None:
            return flag
        selection = options.index(flag.result)
        flag = testors[selection]()
        return flag

    def run(self) -> None:
        for loop in AskFor.LoopTillExit(False):
            loop.set_flag(self._menu_for_operations())
