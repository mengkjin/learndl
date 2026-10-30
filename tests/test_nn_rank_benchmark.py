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

    def test_hybrid_switch_axis_and_boundary(self):
        from src.func import tensor
        for shape,dim in [((4095,2),0),((4096,2),0),((4097,2),0),
                          ((2,4096),-1),((64,64),None)]:
            x=torch.randint(-3,4,shape).float()
            x.reshape(-1)[0]=torch.nan
            length=x.numel() if dim is None else x.shape[dim]
            original=tensor.rank_scatter
            with patch.object(tensor,'rank_scatter',wraps=original) as scatter:
                actual=tensor.rank_hybrid(x,dim=dim,enabled=True,threshold=4096)
                self.assertEqual(scatter.called,length>=4096)
            torch.testing.assert_close(actual,tensor.rank(x,dim=dim),rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(tensor.rank_pct_hybrid(x,dim=dim,enabled=True,threshold=4096),
                                       tensor.rank_pct(x,dim=dim),rtol=0,atol=0,equal_nan=True)
            with patch.object(tensor,'rank_scatter',side_effect=AssertionError('disabled')):
                torch.testing.assert_close(tensor.rank_hybrid(x,dim=dim),tensor.rank(x,dim=dim),rtol=0,atol=0,equal_nan=True)

    def test_hybrid_empty_scalar_and_validation(self):
        from src.func.tensor import rank_hybrid,rank_pct_hybrid,rank,rank_pct
        for x in (torch.tensor(2.),torch.empty(0,3),torch.tensor([float('inf'),float('nan'),1.])):
            for dim in (None,0):
                for new,old in ((rank_hybrid,rank),(rank_pct_hybrid,rank_pct)):
                    torch.testing.assert_close(new(x,dim=dim,enabled=True,threshold=1),old(x,dim=dim),rtol=0,atol=0,equal_nan=True)
        with self.assertRaises(ValueError):rank_hybrid(torch.ones(2),enabled=True,threshold=0)
        for fn in (rank_hybrid,rank_pct_hybrid):
            with self.assertRaises(AssertionError):fn(torch.ones(1,1,1,1),dim=None,enabled=True)
