from __future__ import annotations
from src.proj import Logger , Proj , MACHINE

def rebuild_preprocessed_data() -> None:
    """Archive the model."""

    from src.proj.util.functional.ask import AskFor
    from src.data.preprocess.processors import PrePros

    while True:
        data_keys = PrePros.keys()
        if not data_keys:
            Logger.note('No data keys found for preprocess.')
            return
        flag = AskFor.Options(data_keys , confirm = False , multiple = False , title = f'Which data preprocessor to reconstruct?')
        if flag.no:
            return
        if flag.abort:
            continue
        
        data_key = flag.result
        Logger.note(f'Select [{data_key}] data to reconstruct...')
        flag_type = AskFor.Options(['fit' , 'predict' , 'both'] , confirm = False , multiple = False , title = f'Which type of data to reconstruct? (fit/predict/both)')
        if flag_type.no:
            return
        if flag_type.abort:
            continue
        data_type = flag_type.result
        if MACHINE.platform_coding and data_type != 'predict':
            Logger.alert1('This is a coding machine, skip the fit data reconstruct.')
        else:
            with Proj.vb.temporary_vb('max'):
                if data_type == 'fit':
                    PrePros.get_processor(data_key, type = 'fit').update(reconstruct = True)
                elif data_type == 'predict':
                    PrePros.get_processor(data_key, type = 'predict').update(reconstruct = True)
                elif data_type == 'both':
                    PrePros.get_processor(data_key, type = 'fit').update(reconstruct = True)
                    PrePros.get_processor(data_key, type = 'predict').update(reconstruct = True)
                else:
                    raise ValueError(f'Invalid data type: {data_type}')
        flag = AskFor.Retry(title = f'Do you want to reconstruct more data?')
        if flag.no:
            return