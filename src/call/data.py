from __future__ import annotations
from src.proj import Logger , Proj , MACHINE

def reconstruct_preprocessed_data() -> None:
    """Reconstruct the preprocessed data."""

    from src.proj.util.functional.ask import AskFor
    from src.data.preprocess.processors import PrePros

    data_keys = PrePros.keys()
    if not data_keys:
        Logger.note('No data keys found for preprocess.')
        return

    with Proj.vb.temporary_vb('max'):
        for _ in AskFor.LoopTillExit(message = f'Do you want to reconstruct more data?'):
            flag_key = AskFor.Options(data_keys , confirm = False , multiple = False , title = f'Which data preprocessor to reconstruct?')
            if not flag_key.yes:
                continue
            
            Logger.note(f'Select [{flag_key.result}] data to reconstruct...')
            flag_type = AskFor.Options(['fit' , 'predict' , 'both'] , confirm = False , multiple = False , title = f'Which type of data to reconstruct? (fit/predict/both)')
            if not flag_type.yes:
                continue

            data_type = flag_type.result
            if MACHINE.platform_coding and data_type != 'predict':
                Logger.alert1('This is a coding machine, skip the fit data reconstruct.')
            elif data_type == 'fit':
                PrePros.get_processor(flag_key.result, type = 'fit').update(reconstruct = True)
            elif data_type == 'predict':
                PrePros.get_processor(flag_key.result, type = 'predict').update(reconstruct = True)
            elif data_type == 'both':
                PrePros.get_processor(flag_key.result, type = 'fit').update(reconstruct = True)
                PrePros.get_processor(flag_key.result, type = 'predict').update(reconstruct = True , confirm = False)
            else:
                raise ValueError(f'Invalid data type: {data_type}')