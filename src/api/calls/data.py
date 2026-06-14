"""
Direct calls related to data operations of this project.
"""

from __future__ import annotations
from src.proj import Logger , Proj , MACHINE
from src.api.util import DirectCall

__all__ = ['ReconstructPreprocessedData']

class ReconstructPreprocessedData(DirectCall):
    """Reconstruct the preprocessed data."""
    category = 'Data'
    def run(self) -> None:
        from src.proj.util.functional.ask import AskFor
        from src.data.preprocess.processors import PrePros

        data_keys = PrePros.keys()
        if not data_keys:
            Logger.note('No data keys found for preprocess.')
            return

        with Proj.vb.temporary_vb('max'):
            for loop_flag in AskFor.LoopTillExit(message = f'Do you want to reconstruct more data?'):
                flag_key = AskFor.Options(data_keys , confirm = False , multiple = False , title = f'Which data preprocessor to reconstruct?')
                if not loop_flag.set_flag(flag_key):
                    continue
                
                Logger.note(f'Select [{flag_key.result}] data to reconstruct...')
                flag_type = AskFor.Options(['fit' , 'predict' , 'both'] , confirm = False , multiple = False , title = f'Which type of data to reconstruct? (fit/predict/both)')
                if not loop_flag.set_flag(flag_type):
                    continue

                data_type = flag_type.result
                if MACHINE.platform_coding and data_type != 'predict':
                    Logger.alert1('This is a coding machine, skip the fit data reconstruct.')
                elif data_type == 'fit':
                    PrePros.get_processor(flag_key.result, frame = 'fit').build(reconstruct = True)
                elif data_type == 'predict':
                    PrePros.get_processor(flag_key.result, frame = 'predict').build(reconstruct = True)
                elif data_type == 'both':
                    PrePros.get_processor(flag_key.result, frame = 'fit').build(reconstruct = True)
                    PrePros.get_processor(flag_key.result, frame = 'predict').build(reconstruct = True , confirm = False)
                else:
                    raise ValueError(f'Invalid data type: {data_type}')