"""Callback fastpaths must preserve values, event decisions and call ordering."""
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
from src.res.model.callback.experimental import DEFAULTS, options, latest, basic_group_stats
from src.res.model.callback.fit.early_stop import EarlyStoppage
from src.res.model.callback.fit.retrain import BadAttemptRetrain
from src.res.model.callback.consolidate import ConsolidateCallBack
from src.res.model.util.metric.functions import RankICFunction
from src.res.model.util.core import BatchInput, BatchOutput, BatchData


def bare(cls, enabled=True):
    obj=object.__new__(cls)
    obj.__dict__['callback_fastpath']=DEFAULTS | {k:enabled for k in DEFAULTS if k!='hidden_rank_threshold'}
    return obj


class CallbackFastpathTest(TestCase):
    def test_options_and_latest(self):
        self.assertEqual(options(NS(Param={})),DEFAULTS)
        self.assertTrue(options(NS(Param={'train.experimental_callback_fastpath.dispatch':True}))['dispatch'])
        with self.assertRaises(ValueError): options(NS(Param={'train.experimental_callback_fastpath':{'dispatch':1}}))
        attempt=NS(tables={'valid_epoch_totals':[]})
        self.assertEqual(latest(attempt,'valid','loss'),0.)
        attempt.tables['valid_epoch_totals']=[pd.DataFrame({'total_loss':[i], 'rankic':[i/10]}) for i in range(8)]
        self.assertEqual(latest(attempt,'valid','loss'),7)
        self.assertEqual(latest(attempt,'valid','rankic'),.7)

    def test_hidden_rank_boundaries_and_special_values(self):
        torch.manual_seed(8)
        for n in (1,200,4095,4096,4097,5000):
            for width in (1,32,64):
                h=torch.randn(n,width).round(decimals=1)
                y=torch.randn(n,1)
                for special in (False,True):
                    if special: h[0]=torch.nan; y[-1]=torch.inf
                    data=BatchData(BatchInput(h,y,None,torch.zeros(n,2,dtype=torch.long),torch.ones(n,1,dtype=torch.bool),np.array([1]),np.arange(n)),BatchOutput((y,{'hidden':h})))
                    calc=RankICFunction()
                    for threshold in (4096,5000):
                        torch.testing.assert_close(calc.hidden_rankic(data),calc.hidden_rankic(data,hybrid_threshold=threshold),rtol=0,atol=0,equal_nan=True)
        # Custom metrics are not silently replaced.
        calc.rankic_calculator.calculator=lambda **kw:torch.tensor([.25])
        self.assertEqual(calc.hidden_rankic(data,hybrid_threshold=1).item(),.25)

    def test_basic_values_ties_boundaries_constants(self):
        for n in (1,2,20,100,201):
            rng=np.random.default_rng(n)
            for constant in (False,True):
                df=pd.DataFrame({'pred':np.zeros(n) if constant else rng.integers(-4,5,n),
                    'rtn_lag1_10':rng.normal(size=n),'std_lag1_10':rng.normal(size=n)})
                actual=basic_group_stats(df)
                ranks=df.pred.rank(pct=True);rtn=df.rtn_lag1_10
                z=(rtn-rtn.mean())/(rtn.std()+1e-6)
                expected=pd.Series({'rankic_rtn':df.pred.corr(rtn,method='spearman'),
                    'rankic_std':df.pred.corr(df.std_lag1_10,method='spearman'),
                    'top5pct':z[ranks>=.95].mean(),'mid20pct':z[(ranks>=.4)&(ranks<.6)].mean(),'bot5pct':z[ranks<.05].mean()})
                pd.testing.assert_series_equal(actual,expected,rtol=1e-10,atol=1e-12)

    def test_retrain_local_cache_is_discarded_on_exception(self):
        obj=bare(BadAttemptRetrain)
        metrics=NS(attempt_metrics=NS(best_epoch=Mock(return_value=NS(valid_rankic=.2))))
        with patch.object(BadAttemptRetrain,'metrics',new=property(lambda s:metrics)):
            def check():
                self.assertIs(obj._decision_best_epoch(),obj._decision_best_epoch())
                raise RuntimeError('stop')
            obj._check_retrain=check
            for _ in range(2):
                with self.assertRaises(RuntimeError):obj.on_fit_epoch_end_before()
            self.assertEqual(metrics.attempt_metrics.best_epoch.call_count,2)
            self.assertNotIn('_local_decision_cache',vars(obj))

    def test_early_stop_events_and_no_unused_loss_read(self):
        for mode in ('valid','any'):
            for values in ([1.,1.,1.,1.,1.],[1.,2.,3.,4.,5.],[1.,float('nan'),2.],[]):
                outputs=[]
                for enabled in (False,True):
                    cb=bare(EarlyStoppage,enabled);cb.converge_dataset=mode;cb.converge_patience=5;cb.peak_patience=20
                    cb.peak_epoch=None;cb.peak_epoch_metrics=None
                    metric=Mock();metric.attempt_metrics.latest_epoch.return_value=None
                    metric.compare_epochs.return_value=False
                    metric.attempt_metrics.total_accuracies=values
                    metric.attempt_metrics.total_losses=values
                    status=NS(epoch=4)
                    events=[];cb.add_stop_event=lambda *args:events.append(args)
                    class Status:
                        epoch=4
                        def __getitem__(self,key):return key
                    with patch.object(EarlyStoppage,'metrics',new=property(lambda s:metric)),patch.object(EarlyStoppage,'status',new=property(lambda s:Status())):
                        cb.check_early_stoppage()
                    outputs.append(events)
                self.assertEqual(*outputs)

    def test_dispatch_preserves_custom_enter_exit_and_restart(self):
        traces=[]
        for enabled in (False,True):
            cb=bare(ConsolidateCallBack,enabled);trace=[]
            class Child:
                def __init__(self,name,active):self.name=name;self.active=active;self.logger=Mock()
                def is_hook_implemented(self,h):return h in self.active
                def at_enter(self,h,*a,**k):trace.append((self.name,'enter',h))
                def at_exit(self,h,*a,**k):trace.append((self.name,'exit',h))
            cb.callbacks=[Child('a',{'on_train_batch_end'}),Child('b',set()),Child('c',{'on_train_batch_end'})]
            cb.__dict__['all_hooks']={'on_train_batch_end','on_fit_model_restart'};cb.rebuild_dispatch()
            with patch.object(ConsolidateCallBack,'logger',new=property(lambda s:Mock())),patch.object(ConsolidateCallBack,'status',new=property(lambda s:NS(stage='fit'))):
                cb.at_enter('on_train_batch_end');cb.at_exit('on_train_batch_end')
                cb.callbacks[1].active.add('on_train_batch_end');cb.at_exit('on_fit_model_restart')
                cb.at_enter('on_train_batch_end');cb.at_exit('on_train_batch_end')
            traces.append(trace)
        self.assertEqual(*traces)

    def test_retrain_decisions_boundaries_and_nan(self):
        for ep,ic,global_ic,nan,remain,attempt,ended in [
            (10,.5,.5,False,4,0,True),(11,.09,.2,False,4,0,True),
            (20,.1,.2,False,4,0,True),(21,.04,.04,False,4,0,True),
            (21,.05,.04,False,4,0,True),(21,.01,.1,False,4,0,True),
            (21,.01,.01,False,4,4,True),(21,.01,.01,False,4,0,False),
            (None,.1,.1,False,4,0,True),(12,.1,.1,True,1,0,False),
            (12,.1,.1,True,0,0,False)]:
            results=[]
            for enabled in (False,True):
                cb=bare(BadAttemptRetrain,enabled)
                cb.early_exit=10;cb.moderate_early_exit=20;cb.moderate_min_ic=.1;cb.min_ic=.05
                cb.max_attempt=4;cb.lr_multiplier=[1,.3,.1,.03,.01];cb.remain_nan_redo=remain;cb.is_nanloss=nan
                best=NS(valid_rankic=ic)
                metrics=NS(attempt_metrics=NS(best_epoch=Mock(return_value=best)),model_metrics=NS(best_ic=global_ic))
                status=NS(end_attempt_event=NS(effective_epoch=ep) if ep is not None else None,attempt=attempt,loop_end=ended)
                events=[];cb.trigger_retrain=lambda *args:events.append(args)
                with patch.object(BadAttemptRetrain,'metrics',new=property(lambda s:metrics)),patch.object(BadAttemptRetrain,'status',new=property(lambda s:status)),patch.object(BadAttemptRetrain,'texts',new=property(lambda s:NS(attempt_key='a'))),patch.object(BadAttemptRetrain,'model',new=property(lambda s:NS(param_has_nan=True))),patch.object(BadAttemptRetrain,'logger',new=property(lambda s:Mock())):
                    try:cb.on_fit_epoch_end_before();error=None
                    except Exception as exc:error=str(exc)
                results.append((events,error,cb.remain_nan_redo))
                self.assertNotIn('_local_decision_cache',vars(cb))
            self.assertEqual(*results)

    def test_basic_actual_callback_resume_and_nan_rows(self):
        from src.res.model.callback.test.basic import BasicTestResult
        import src.res.model.callback.test.basic as module
        rng=np.random.default_rng(13)
        data=pd.DataFrame({'model_num':np.repeat([0,1],100),'model_date':20200101,'submodel':'best',
            'date':np.tile(np.repeat([1,2],50),2),'pred':rng.integers(-3,4,200).astype(float),
            'rtn_lag1_10':rng.normal(size=200),'std_lag1_10':rng.normal(size=200)})
        data.loc[0,'pred']=np.nan;data.loc[1,'rtn_lag1_10']=np.nan
        outputs=[]
        for enabled in (False,True):
            cb=NS(EVAL_LABELS=BasicTestResult.EVAL_LABELS,config=NS(is_resuming=False),path_test_df='unused',test_df_cols=['model_num','model_date','submodel','date',*BasicTestResult._stat_cols],stat_cols=list(BasicTestResult._stat_cols),test_full_dates=np.array([1,2]),record=NS(get_preds=lambda dates:data[data.date.isin(dates)].copy()),_attach_eval_labels=lambda x:x,callback_fastpath=DEFAULTS | {'basic_test':enabled},vb_level=0,indent=0)
            with patch.object(module.Save,'df'):
                full=BasicTestResult.complete_test_df(cb)
                cb.config.is_resuming=True
                saved=full[full.date==1]
                with patch.object(module.Load,'df',return_value=saved):
                    resumed=BasicTestResult.complete_test_df(cb)
            pd.testing.assert_frame_equal(full.sort_values(['model_num','date']).reset_index(drop=True),resumed,rtol=1e-10,atol=1e-12)
            outputs.append(full)
        pd.testing.assert_frame_equal(*outputs,rtol=1e-10,atol=1e-12)

    def test_summary_full_output_and_empty_hidden(self):
        from src.res.model.callback.monitor.summary import SummaryWriter
        from src.res.model.diagnostics.callback_benchmark import DecisionFixture
        class Writer:
            def __init__(self):self.calls=[]
            def add_scalars(self,*args):self.calls.append(args)
        outputs=[]
        for enabled in (False,True):
            attempt=DecisionFixture(False).attempt
            attempt.tables['train_epoch_totals']=attempt.tables['valid_epoch_totals']
            for frame in attempt.tables['valid_epoch_totals']:frame['rankic']=.2
            cb=NS(TSBOARD_PREFIXIS=SummaryWriter.TSBOARD_PREFIXIS,metrics=NS(attempt_metrics=attempt),writer=Writer(),step_epoch=7,callback_fastpath=DEFAULTS | {'summary_latest':enabled})
            SummaryWriter.add_total_metric(cb);outputs.append(cb.writer.calls)
            cb.HIDDEN_FEATURE_MODE=True;cb.batch_data=NS(output=NS(other={}))
            SummaryWriter.append_batch_hidden_summary(cb)
        self.assertEqual(*outputs)

    def test_profile_restores_methods(self):
        from src.res.model.diagnostics.callback_profile import CallbackProfile
        from src.res.model.callback.monitor.summary import SummaryWriter
        original=SummaryWriter.add_total_metric
        profile=CallbackProfile()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with profile:
                self.assertIsNot(SummaryWriter.add_total_metric,original)
                torch.corrcoef(torch.randn(3,20))
                raise RuntimeError('injected')
        self.assertIs(SummaryWriter.add_total_metric,original)
        self.assertEqual(profile.stats['torch.corrcoef']['calls'],1)

    def test_valid_early_stop_skips_loss_and_preserves_history_scale(self):
        from src.res.model.callback.fit.early_stop import arr_converge
        # Last-five-only normalization would change the answer for this sequence.
        history=[10000.]+[1.,1.01,1.02,1.03,1.04]
        self.assertTrue(arr_converge(history,5))
        self.assertFalse(arr_converge(history[-5:],5))
        cb=bare(EarlyStoppage);cb.converge_dataset='valid';cb.converge_patience=5;cb.peak_patience=20
        cb.peak_epoch=None;cb.peak_epoch_metrics=None;events=[];cb.add_stop_event=lambda *a:events.append(a)
        class Attempt:
            total_accuracies=history
            def latest_epoch(self):return None
            @property
            def total_losses(self):raise AssertionError('unused loss read')
        class Status:
            def __getitem__(self,key):return key
        with patch.object(EarlyStoppage,'metrics',new=property(lambda s:NS(attempt_metrics=Attempt(),compare_epochs=lambda *a:False))),patch.object(EarlyStoppage,'status',new=property(lambda s:Status())):
            cb.check_early_stoppage()
        self.assertEqual(events,[('Valid Converged',-5)])

    def test_dispatch_exception_and_filtering(self):
        from src.res.model.diagnostics.callback_benchmark import DispatchFixture
        for enabled in (False,True):
            trace=[];cb=DispatchFixture(enabled,trace)
            def fail(h):raise RuntimeError('callback failure')
            cb.callbacks[3].at_exit=fail
            with self.assertRaisesRegex(RuntimeError,'callback failure'):cb.at_exit('on_train_batch_end')
            self.assertEqual(trace,[(0,'exit','on_train_batch_end')])
        cb=bare(ConsolidateCallBack)
        class A:
            OverrideCallbacks=[];ConflictModuleTypes=[];ConflictModuleNames=[];ConflictCallbacks=[]
        class B(A):OverrideCallbacks=['A']
        with patch.object(ConsolidateCallBack,'config',new=property(lambda s:NS(module_type='nn',model_module='gru'))),patch.object(ConsolidateCallBack,'logger',new=property(lambda s:Mock())):
            remaining=cb.resolving_callbacks([A(),B()])
        self.assertEqual([type(x).__name__ for x in remaining],['B'])

    def test_profile_async_export_completion(self):
        from concurrent.futures import Future
        from src.res.model.diagnostics.callback_profile import CallbackProfile
        from src.proj import Save
        f=Future()
        with patch.object(Save,'dfs',return_value=f):
            with CallbackProfile() as profile:
                Save.dfs({},'dummy.xlsx',async_save=True)
                f.set_result(True)
            self.assertTrue(profile.exports[0]['asynchronous'])
            self.assertIsNone(profile.exports[0]['error'])
            self.assertGreaterEqual(profile.exports[0]['completion_latency_after_submit_seconds'],0.)

    def test_benchmark_failure_mails_closed_log_and_preserves_failure(self):
        import tempfile, json
        from pathlib import Path
        from src.res.model.diagnostics import callback_benchmark as bench
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run';sender=Mock(return_value=True)
            with patch.object(bench.sys,'argv',['bench','--output',str(output)]),patch.object(bench,'run_micro',side_effect=ValueError('injected validation failure')),patch('src.proj.util.web.task_mail.TaskResultMail',return_value=sender):
                with self.assertRaisesRegex(ValueError,'injected validation failure'):bench.main()
            sender.assert_called_once()
            self.assertEqual(json.loads((output/'report.json').read_text())['status'],'failed')
            self.assertIn('injected validation failure',(output/'console.log').read_text())
            attachments=sender.call_args.kwargs['attachments']
            self.assertEqual({p.suffix for p in attachments},{'.json','.csv','.log','.zip'})
            with patch('src.proj.util.web.task_mail.TaskResultMail',return_value=Mock(return_value=False)):
                with self.assertRaisesRegex(RuntimeError,'retry only delivery'):bench.send_results(NS(output=output,recipient=None),{})

    def test_short_suite_keeps_four_runs_after_child_failure(self):
        import tempfile
        from pathlib import Path
        from src.res.model.diagnostics import callback_benchmark as bench
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);calls=[]
            def launch(command,**kwargs):
                if '--single' in command:
                    calls.append(command)
                    return NS(returncode=1 if len(calls)==2 else 0)
                return NS(returncode=0)
            report={'rows':[],'eligible_flags':['summary_latest']}
            with patch.object(bench.subprocess,'run',side_effect=launch),patch.object(bench.shutil,'copytree'),patch.object(bench.shutil,'copy2'):
                with self.assertRaisesRegex(AssertionError,'short tests failed'):
                    bench.run_short(NS(output=output,threshold=5000),report)
            self.assertEqual([c[c.index('--flags')+1] for c in calls],['','summary_latest','summary_latest',''])
            self.assertEqual(len(report['short_tests']),4)
            self.assertTrue(all('process_seconds' in r for r in report['short_tests']))
