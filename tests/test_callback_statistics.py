"""Regression checks for integrated callback statistics and measurement tools."""
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
from src.res.model.callback.test.basic import basic_group_stats, BasicTestResult
from src.res.model.util.metric.stage_metrics import AttemptMetrics
from src.res.model.util.metric.functions import RankICFunction
from src.res.model.util.core import BatchInput, BatchOutput, BatchData
from src.res.model.diagnostics.callback_benchmark import latest_reference, basic_legacy
from src.func.metric import spearman


class CallbackStatisticsTest(TestCase):
    def test_latest_without_history_scan(self):
        attempt=object.__new__(AttemptMetrics)
        for count in (0,1,8):
            attempt.tables={f'{ds}_epoch_totals':[pd.DataFrame({'total_loss':[i], 'total_accuracy':[i/2], 'rankic':[i/10]}) for i in range(count)] for ds in ('train','valid')}
            for ds in ('train','valid'):
                for metric in ('loss','accuracy','rankic'):
                    expected=latest_reference(attempt,ds,metric)
                    with patch.object(attempt,'total_metrics',side_effect=AssertionError('history scan')):
                        self.assertEqual(attempt.latest(ds,metric),expected)

    def test_hidden_rank_boundaries_and_special_values(self):
        torch.manual_seed(8)
        for n in (1,200,4095,4096,4097,5000):
            for width in (1,32,64):
                h=torch.randn(n,width).round(decimals=1);y=torch.randn(n,1)
                for special in (False,True):
                    if special:h[0]=torch.nan;y[-1]=torch.inf
                    data=BatchData(BatchInput(h,y,None,torch.zeros(n,2,dtype=torch.long),torch.ones(n,1,dtype=torch.bool),np.array([1]),np.arange(n)),BatchOutput((y,{'hidden':h})))
                    calc=RankICFunction();reference=RankICFunction()
                    reference.rankic_calculator.calculator=lambda pred,label,weight=None,dim=None,**kw:spearman(pred,label,weight,dim,scatter_enabled=False,**kw)
                    torch.testing.assert_close(calc.hidden_rankic(data),reference.hidden_rankic(data),rtol=0,atol=0,equal_nan=True)
        calc.rankic_calculator.calculator=lambda **kw:torch.tensor([.25])
        self.assertEqual(calc.hidden_rankic(data).item(),.25)

    def test_summary_output_and_empty_hidden(self):
        from src.res.model.callback.monitor.summary import SummaryWriter
        attempt=object.__new__(AttemptMetrics)
        for count in (0,3):
            attempt.tables={f'{ds}_epoch_totals':[pd.DataFrame({'total_loss':[i], 'total_accuracy':[i/2], 'rankic':[i/10]}) for i in range(count)] for ds in ('train','valid')}
            cb=NS(TSBOARD_PREFIXIS=SummaryWriter.TSBOARD_PREFIXIS,metrics=NS(attempt_metrics=attempt),writer=Mock(),step_epoch=7)
            SummaryWriter.add_total_metric(cb)
            for call,metric in zip(cb.writer.add_scalars.call_args_list,('accuracy','loss','rankic')):
                self.assertEqual(call.args,(f'00.Metrics/Total{metric.title()}',{ds:latest_reference(attempt,ds,metric) for ds in ('train','valid')},7))
            cb.HIDDEN_FEATURE_MODE=True;cb.batch_data=NS(output=NS(other={}))
            SummaryWriter.append_batch_hidden_summary(cb)

    def test_basic_actual_callback_resume_and_nan_rows(self):
        import src.res.model.callback.test.basic as module
        rng=np.random.default_rng(13)
        data=pd.DataFrame({'model_num':np.repeat([0,1],100),'model_date':20200101,'submodel':'best',
            'date':np.tile(np.repeat([1,2],50),2),'pred':rng.integers(-3,4,200).astype(float),
            'rtn_lag1_10':rng.normal(size=200),'std_lag1_10':rng.normal(size=200)})
        data.loc[0,'pred']=np.nan;data.loc[1,'rtn_lag1_10']=np.nan
        cb=NS(EVAL_LABELS=BasicTestResult.EVAL_LABELS,config=NS(is_resuming=False),path_test_df='unused',test_df_cols=['model_num','model_date','submodel','date',*BasicTestResult._stat_cols],stat_cols=list(BasicTestResult._stat_cols),test_full_dates=np.array([1,2]),record=NS(get_preds=lambda dates:data[data.date.isin(dates)].copy()),_attach_eval_labels=lambda x:x,vb_level=0,indent=0)
        with patch.object(module.Save,'df'):
            full=BasicTestResult.complete_test_df(cb)
            expected=basic_legacy(data.dropna()).reset_index()
            pd.testing.assert_frame_equal(full,expected,rtol=1e-10,atol=1e-12)
            cb.config.is_resuming=True
            with patch.object(module.Load,'df',return_value=full[full.date==1]):
                resumed=BasicTestResult.complete_test_df(cb)
            pd.testing.assert_frame_equal(full.sort_values(['model_num','date']).reset_index(drop=True),resumed,rtol=1e-10,atol=1e-12)

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
                report={'rows':[]}
                with patch.object(bench.subprocess,'run',side_effect=launch),patch.object(bench.shutil,'copytree'),patch.object(bench.shutil,'copy2'):
                    with self.assertRaisesRegex(AssertionError,'short tests failed'):
                        bench.run_short(NS(output=output),report)
                self.assertEqual(len(calls),4)
                self.assertTrue(all('--flags' not in c for c in calls))
                self.assertEqual(len(report['short_tests']),4)
                self.assertTrue(all('process_seconds' in r for r in report['short_tests']))


    def test_dispatch_uses_live_order_and_propagates_exception(self):
        from src.res.model.callback.consolidate import ConsolidateCallBack
        cb=object.__new__(ConsolidateCallBack);trace=[]
        children=[NS(logger=Mock(),is_hook_implemented=Mock(return_value=i != 1),at_enter=Mock(side_effect=lambda h,i=i:trace.append(i))) for i in range(3)]
        cb.callbacks=children
        with patch.object(ConsolidateCallBack,'logger',new=property(lambda s:Mock())),patch.object(ConsolidateCallBack,'status',new=property(lambda s:NS(stage='fit'))):
            cb.at_enter('on_train_batch_end')
            self.assertEqual(trace,[0,2])
            children[1].is_hook_implemented.return_value=True
            children[1].at_enter.side_effect=RuntimeError('callback failure')
            trace.clear()
            with self.assertRaisesRegex(RuntimeError,'callback failure'):cb.at_enter('on_train_batch_end')
            self.assertEqual(trace,[0])

    def test_retrain_reads_best_without_cache(self):
        from src.res.model.callback.fit.retrain import BadAttemptRetrain
        cb=object.__new__(BadAttemptRetrain);cb.min_ic=.05
        best=Mock(side_effect=[NS(valid_rankic=.2),NS(valid_rankic=.01)])
        metric=NS(attempt_metrics=NS(best_epoch=best),model_metrics=NS(best_ic=.01))
        with patch.object(BadAttemptRetrain,'metrics',new=property(lambda s:metric)):
            self.assertEqual(cb._attempt_best_ic,.2)
            self.assertTrue(cb.is_low_ic)
        self.assertEqual(best.call_count,2)

    def test_valid_early_stop_skips_loss_and_preserves_history_scale(self):
            from src.res.model.callback.fit.early_stop import arr_converge, EarlyStoppage
            # Last-five-only normalization would change the answer for this sequence.
            history=[10000.]+[1.,1.01,1.02,1.03,1.04]
            self.assertTrue(arr_converge(history,5))
            self.assertFalse(arr_converge(history[-5:],5))
            cb=object.__new__(EarlyStoppage);cb.converge_dataset='valid';cb.converge_patience=5;cb.peak_patience=20
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
