"""Frozen rolling splits used by the ordinary model trainer."""
import numpy as np


def validate_folds(folds, dates):
    dates=np.asarray(dates)
    model_dates=[int(f['model_date']) for f in folds]
    test_dates=[int(d) for f in folds for d in f['test']]
    if model_dates!=sorted(set(model_dates)) or test_dates!=sorted(set(test_dates)):
        raise ValueError('Rolling plan dates must be ordered and unique')
    if test_dates!=dates[1:].tolist() or not set(model_dates)<=set(dates.tolist()):
        raise ValueError('Rolling plan differs from the loaded test calendar')
    for index,fold in enumerate(folds):
        if not fold['train'] or not fold['valid'] or not fold['test']:
            raise ValueError('Empty rolling split')
        if not max(fold['train'])<min(fold['valid'])<=max(fold['valid'])<=fold['model_date']<min(fold['test']):
            raise ValueError('Invalid rolling split boundaries')
        end=model_dates[index+1] if index+1<len(folds) else int(dates[-1])
        expected=dates[(dates>fold['model_date']) & (dates<=end)].tolist()
        if fold['test']!=expected:
            raise ValueError('Rolling test dates differ from trainer interval')
    return np.asarray(model_dates),np.asarray(test_dates)


def select_fold(folds, model_date):
    selected=[f for f in folds if int(f['model_date'])==int(model_date)]
    if len(selected)!=1:
        raise ValueError(f'Expected one frozen split for model date {model_date}')
    return selected[0]
