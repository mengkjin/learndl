"""Temporary, benchmark-only callback instrumentation; no CUDA synchronization.

Times are host inclusive/exclusive durations, not GPU kernel durations. Nested
work must not be summed twice. Patches are restored even after failed training.
"""
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from time import perf_counter
from unittest.mock import patch
import inspect


class CallbackProfile:
    def __init__(self):
        self.stats = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
        self.stack = []
        self.events = []
        self.diagnostics = []
        self.active = []
        self.epochs = []
        self.exports = []
        self.epoch_start = None
        self.patches = ExitStack()

    def wrap(self, owner, name, label=None, capture=None):
        original = getattr(owner, name)
        label = label or f'{owner.__name__}.{name}'
        @wraps(original)
        def measured(*args, **kwargs):
            start = perf_counter(); frame = [0.]; self.stack.append(frame)
            try:
                result = original(*args, **kwargs)
            finally:
                elapsed = perf_counter()-start
                self.stack.pop()
                if self.stack:
                    self.stack[-1][0] += elapsed
                row = self.stats[label]
                row['calls'] += 1
                row['inclusive_seconds'] += elapsed
                row['exclusive_seconds'] += elapsed-frame[0]
            if capture:
                capture(args, kwargs, result)
            return result
        descriptor = inspect.getattr_static(owner, name)
        if isinstance(descriptor, staticmethod):
            measured = staticmethod(measured)
        elif isinstance(descriptor, classmethod):
            # original is already bound to the class.
            wrapped = measured
            measured = classmethod(lambda cls, *a, **k: wrapped(*a, **k))
        self.patches.enter_context(patch.object(owner, name, measured))

    def __enter__(self):
        try:
            return self._install()
        except BaseException:
            self.patches.close()
            raise

    def _install(self):
        from src.res.model.callback.consolidate import ConsolidateCallBack
        from src.res.model.callback import monitor, fit, test
        from src.res.model.callback.monitor.summary import SummaryWriter
        from src.res.model.callback.test.detail import DetailedAlphaAnalysis
        from src.res.model.callback.test.basic import BasicTestResult
        from src.res.model.util.metric.functions import RankICFunction
        from src.res.model.util.trainer.status import TrainerStatus
        from torch.utils.tensorboard import SummaryWriter as TorchWriter
        from src.proj import Save
        def active(args, kwargs, result):
            self.active = [type(cb).__name__ for cb in args[0].callbacks]
        self.wrap(ConsolidateCallBack, 'init_callbacks', capture=active)
        def epoch_capture(args, kwargs, result):
            cb, hook = args[:2]
            if hook == 'on_fit_epoch_start':
                self.epoch_start = perf_counter()
            elif hook == 'on_validation_epoch_end' and self.epoch_start is not None:
                self.epochs.append(dict(epoch=cb.status.epoch, phase=cb.status.phase,
                    attempt=cb.status.attempt, diagnostic_epoch=cb.status.epoch % SummaryWriter.HIDDEN_FEATURE_STEP == 0,
                    train_valid_host_seconds=perf_counter()-self.epoch_start))
                self.epoch_start = None
        self.wrap(ConsolidateCallBack, 'at_enter')
        self.wrap(ConsolidateCallBack, 'at_exit', capture=epoch_capture)
        seen = set()
        for module in (monitor,fit,test):
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls in seen: continue
                seen.add(cls)
                for name, value in vars(cls).items():
                    if name.startswith('on_') and callable(value): self.wrap(cls,name)
        for name in ('append_batch_hidden_summary','add_hidden_summaries','add_total_metric','sync_latest_tensorboard'):
            self.wrap(SummaryWriter,name)
        self.wrap(RankICFunction,'hidden_rankic')
        # corrcoef is invoked by hidden diagnostic; CUDA completion is deferred.
        import torch
        self.wrap(torch,'corrcoef',label='torch.corrcoef')
        for name in ('add_scalar','add_scalars','add_text','add_histogram','add_image','flush','close'):
            self.wrap(TorchWriter,name)
        for name in ('add_hidden_rankic','add_hidden_correlation'):
            def capture(args, kwargs, result, name=name):
                cb, value = args[:2]
                self.diagnostics.append(dict(kind=name, dataset=cb.status.dataset,
                    epoch=cb.status.epoch, values=value.detach().cpu().tolist()))
            self.wrap(SummaryWriter,name,capture=capture)
        self.wrap(BasicTestResult,'complete_test_df')
        for name in ('factor_test','fmp_test','get_factor','display_export'):
            self.wrap(DetailedAlphaAnalysis,name)
        from src.res.factor.analytic import test_basics
        for _, cls in inspect.getmembers(test_basics,inspect.isclass):
            if 'plot' in vars(cls): self.wrap(cls,'plot')
        def exported(args, kwargs, result):
            path = kwargs.get('path', args[1] if len(args)>1 else '')
            row = dict(path=str(path), asynchronous=hasattr(result, 'add_done_callback'))
            self.exports.append(row)
            submitted = perf_counter()
            if row['asynchronous']:
                def done(future):
                    row['completion_latency_after_submit_seconds'] = perf_counter()-submitted
                    row['error'] = str(future.exception()) if future.exception() else None
                result.add_done_callback(done)
            else:
                row['completion_latency_after_submit_seconds'] = 0.
        for name in ('dfs','figs'):
            self.wrap(Save,name,label=f'Save.{name}',capture=exported)
        self.wrap(Save,'async_wait_all',label='Save.async_wait_all')
        def events(args, kwargs, result):
            self.events.append(dict(args=[str(x) for x in args[1:]], kwargs={k:str(v) for k,v in kwargs.items()}))
        self.wrap(TrainerStatus,'add_epoch_event',capture=events)
        return self

    def report(self):
        return dict(active_callbacks=self.active, timings=dict(self.stats), events=self.events,
                    diagnostics=self.diagnostics, epochs=self.epochs, exports=self.exports,
                    timing_semantics='Host inclusive/exclusive seconds; no per-hook CUDA synchronization. Diagnostic capture copies only logged summaries to CPU after timed method.')

    def __exit__(self,*exc):
        return self.patches.__exit__(*exc)
