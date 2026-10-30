"""Benchmark mail owns delivery after logs close, including failed runs."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch, Mock

import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rank_benchmark',ROOT/'scripts/5_test/benchmark_nn_rank_scatter.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class RankBenchmarkTest(TestCase):
    def setUp(self):
        enabled = patch('src.proj.env.MACHINE.emailable', True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_disabled_machine_skips_without_reading_results(self):
        with patch('src.proj.env.MACHINE.emailable', False), \
             patch('src.proj.util.web.task_mail.TaskResultMail') as sender:
            bench.send_results(SimpleNamespace(output=Path('/missing/report.json'), recipient=None))
            sender.assert_not_called()

    def test_success_sends_closed_log_and_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.json'
            sender=Mock(return_value=True)
            def execute(args):
                bench.save_report(args.output,{'status':'success','rows':[]})
                args.output.with_suffix('.csv').write_text('header\n')
                print('completed marker')
            with patch.object(bench.sys,'argv',['benchmark','--output',str(output),'--recipient','test@example.com']), \
                 patch.object(bench,'execute',side_effect=execute), \
                 patch('src.proj.util.web.task_mail.TaskResultMail',return_value=sender):
                bench.main()
            sender.assert_called_once()
            self.assertEqual(sender.call_args.args[2],'test@example.com')
            self.assertEqual({p.suffix for p in sender.call_args.kwargs['attachments']},{'.json','.csv','.log'})
            self.assertIn('completed marker',output.with_suffix('.log').read_text())

    def test_failure_still_mails_and_raises_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.json'
            sender=Mock(return_value=True)
            with patch.object(bench.sys,'argv',['benchmark','--output',str(output)]), \
                 patch.object(bench,'execute',side_effect=ValueError('simulated failure')), \
                 patch('src.proj.util.web.task_mail.TaskResultMail',return_value=sender):
                with self.assertRaisesRegex(ValueError,'simulated failure'): bench.main()
            sender.assert_called_once()
            self.assertEqual(json.loads(output.read_text())['status'],'failed')
            self.assertIn('simulated failure',output.with_suffix('.log').read_text())

    def test_mail_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.json'
            bench.save_report(output,{'status':'success','rows':[]})
            with patch('src.proj.util.web.task_mail.TaskResultMail',return_value=Mock(return_value=False)):
                with self.assertRaisesRegex(RuntimeError,'retry only delivery'):
                    bench.send_results(SimpleNamespace(output=output,recipient=None))

    def test_scenarios_cover_axis_layout_and_precision(self):
        cases=list(bench.scenarios(torch.device('cpu'),[1500,3000,4000],'full'))
        self.assertTrue(any(x.ndim==3 for _,x,_,_ in cases))
        self.assertTrue(any(not x.is_contiguous() for _,x,_,_ in cases))
        self.assertTrue(any(c['dim']==-1 for c,_,_,_ in cases))
        self.assertTrue(any(c['dim'] is None for c,_,_,_ in cases))
        self.assertTrue(any(w is not None for _,_,_,w in cases))
        self.assertTrue(any(x.dtype==torch.float16 for _,x,_,_ in cases))

    def test_spearman_dispatch_and_legacy_switch(self):
        from src.func import metric
        for shape,dim in [((4999,1),0),((5000,1),0),((1,5000),-1),((100,50),None)]:
            x=torch.randn(shape); y=torch.randn(shape)
            x=x.round(decimals=1); y=y.round(decimals=1)
            expected=metric.spearman(x,y,dim=dim,scatter_enabled=False)
            with patch.object(metric,'rank_scatter',wraps=metric.rank_scatter) as scatter:
                actual=metric.spearman(x,y,dim=dim)
                length=x.numel() if dim is None else x.shape[dim]
                self.assertEqual(scatter.called,length>=5000)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
            with patch.object(metric,'rank_scatter',side_effect=AssertionError('disabled')):
                metric.spearman(x,y,dim=dim,scatter_enabled=False)

    def test_scatter_percentiles_preserve_legacy(self):
        from src.func.tensor import rank_pct,rank_pct_scatter
        for x in (torch.tensor(2.),torch.empty(0,3),torch.tensor([[1.,float('nan')],[1.,float('inf')]]).T):
            for dim in (None,0,-1):
                torch.testing.assert_close(rank_pct_scatter(x,dim=dim),rank_pct(x,dim=dim),rtol=0,atol=0,equal_nan=True)
