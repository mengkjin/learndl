"""
Direct calls related to data operations of this project.
"""

from __future__ import annotations
from src.proj import Logger , Proj , MACHINE
from src.api.util.direct_call import DirectCall

__all__ = ['ReconstructPreprocessedData' , 'RecalculateHistNorm']

class ReconstructPreprocessedData(DirectCall):
    """Reconstruct the preprocessed data."""
    category = 'Data'
    def run(self) -> None:
        from src.proj.util.cli import AskFor
        from src.data.preprocess.processors import PrePros

        data_keys = PrePros.keys()
        if not data_keys:
            Logger.note('No data keys found for preprocess.')
            return

        with Proj.vb.temporary_vb('max'):
            for loop in AskFor.LoopTillExit(message = f'Do you want to reconstruct more data?'):
                flag_key = AskFor.Options(
                    data_keys , confirm = False , multiple = False , allow_back = False ,
                    title = f'Which data preprocessor to reconstruct?',
                    help_description='Pick a registered PrePros processor key; rebuilds feather/parquet under data/.',
                )
                if not loop.set_flag(flag_key) or flag_key.result is None:
                    continue
                
                Logger.note(f'Select [{flag_key.result}] data to reconstruct...')
                flag_type = AskFor.Options(
                    ['fit' , 'predict' , 'both'] , confirm = False , multiple = False ,
                    title = f'Which type of data to reconstruct? (fit/predict/both)',
                    help_description='fit: training samples; predict: inference features; both: rebuild both frames.',
                    option_help={
                        'fit': 'Rebuild fit frame used for model training.',
                        'predict': 'Rebuild predict frame used at inference time.',
                        'both': 'Rebuild fit then predict (fit skipped on coding machines).',
                    },
                )
                if not loop.set_flag(flag_type) or flag_type.result is None:
                    continue
                data_type = flag_type.result
                if MACHINE.platform_coding and flag_type.result != 'predict':
                    Logger.alert1('This is a coding machine, fit data reconstruction costs a lot of time.')
                if flag_type.result == 'fit':
                    PrePros.get_processor(flag_key.result, frame = 'fit').build(reconstruct = True)
                elif flag_type.result == 'predict':
                    PrePros.get_processor(flag_key.result, frame = 'predict').build(reconstruct = True)
                elif flag_type.result == 'both':
                    PrePros.get_processor(flag_key.result, frame = 'fit').build(reconstruct = True)
                    PrePros.get_processor(flag_key.result, frame = 'predict').build(reconstruct = True , confirm = False)
                else:
                    raise ValueError(f'Invalid data type: {data_type}')

class RecalculateHistNorm(DirectCall):
    """Recalculate the historical normalisation statistics."""
    category = 'Data'
    def run(self) -> None:
        from src.data.preprocess.hist_norm import PreProHistNorm
        PreProHistNorm.recalculate_all()