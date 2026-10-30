"""Independent NN experiments: numerical gates, index semantics and delivery lifecycle."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/5_test'
sys.path.insert(0,str(SCRIPTS))
import _nn_benchmark_common as common
import benchmark_nn_hidden_corr as hidden
import benchmark_nn_storage_index as storage


class HiddenCorrExperimentTests(TestCase):
    def test_random_float64_value_and_gradient(self):
        torch.manual_seed(71)
        checks=hidden.check_case(torch.randn(83,32,dtype=torch.float64),'random')
        self.assertTrue(all(r['passed'] for r in checks),checks)

    def test_constant_and_container_inputs(self):
        for value in (torch.ones(12,4,dtype=torch.float64),torch.randn(12,4,dtype=torch.float64)):
            for container in (None,list,tuple):
                checks=hidden.check_case(value,'constant/container',container=container)
                self.assertTrue(all(r['passed'] for r in checks),checks)

    def test_gru_gradients_and_update(self):
        checks=hidden.validate_network(torch.device('cpu'),torch.float64)
        self.assertTrue(all(r['passed'] for r in checks),checks)

    def test_normal_nonfinite_is_failure(self):
        for value in (float('nan'),float('inf')):
            result=hidden.compare(torch.tensor(value),torch.tensor(value),torch.float32)
            self.assertFalse(result['passed'])
            self.assertEqual(result['failed_count'],1)
            self.assertEqual(result['first_failed_indices'],[[]])

    def test_error_locations_and_fixed_tolerance(self):
        result=hidden.compare(torch.tensor([0.,2.]),torch.tensor([0.,1.]),torch.float32)
        self.assertFalse(result['passed'])
        self.assertEqual(result['first_failed_indices'],[[1]])
        self.assertEqual((result['rtol'],result['atol']),(1e-5,1e-6))

    def test_extreme_samples_and_specials(self):
        for n in (0,1):
            checks=hidden.check_case(torch.zeros(n,4,dtype=torch.float64),'tiny',exceptional=True)
            self.assertTrue(all(r['passed'] for r in checks),checks)
        value=torch.randn(12,4,dtype=torch.float64);value[0,0]=float('inf')
        checks=hidden.check_case(value,'inf',exceptional=True)
        self.assertTrue(all(r['passed'] for r in checks),checks)

    def test_faster_inaccurate_candidate_is_rejected(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            args=SimpleNamespace(device='cpu',sizes=[8],hidden_dims=[4],repeats=3,
                                 output=Path(root)/'gate.json')
            report={'rows':[],'validation':[]}
            functions={'legacy':hidden.legacy,'wrong':lambda x:hidden.legacy(x)+1}
            with patch.object(hidden,'FUNCTIONS',functions), \
                 patch.object(hidden,'validation_cases',return_value=[]), \
                 patch.object(hidden,'validate_network',return_value=[]), \
                 patch.object(hidden,'timed_variants',return_value=({'legacy':[1.]*3,'wrong':[.1]*3},1)):
                with self.assertRaisesRegex(RuntimeError,'Candidate validation failed'):
                    hidden.execute(args,report)
            self.assertFalse(report['validation_summary']['wrong'])
            self.assertTrue(all(not row['qualified'] for row in report['rows'] if row['implementation']=='wrong'))
            self.assertTrue(args.output.exists())

    def test_backward_never_accumulates(self):
        x=torch.randn(30,4,requires_grad=True)
        for _ in range(3):hidden.timed_call(hidden.joint_stats,x,True)
        self.assertIsNone(x.grad)


class StorageExperimentTests(TestCase):
    def test_operation_sequences_memory_and_disk(self):
        with tempfile.TemporaryDirectory() as root:
            checks=storage.validate(root)
            self.assertEqual(len(checks),28)
            self.assertTrue(all(c['passed'] for c in checks))

    def test_record_snapshot_and_overwrite_order(self):
        with tempfile.TemporaryDirectory() as root:
            store=storage.DictStorage(True,root)
            store.save(None,'a','g');store.save(None,'b','g')
            old=store.records
            store.save(None,'a','new')
            self.assertEqual(store.records['key'].tolist(),['b','a'])
            self.assertEqual(old['key'].tolist(),['a','b'])
            self.assertEqual(list(store.groups['g']),['b'])
            with self.assertRaises(KeyError):store.del_one('absent')

    def test_timer_and_memory_fields(self):
        with tempfile.TemporaryDirectory() as root:
            samples=storage.measure_operation(root,'metadata','add',4,3)
            self.assertEqual(set(samples),{'legacy','dict'})
            self.assertTrue(all(len(v)==3 and min(v)>0 for v in samples.values()))
            memory=storage.index_memory(root,4)
            self.assertTrue(all(v['traced_live_bytes']>0 for v in memory.values()))


class CommonExperimentTests(TestCase):
    def setUp(self):
        enabled = patch('src.proj.env.MACHINE.emailable', True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def invoke(self,execute,root,sender,no_email=False):
        argv=['--output',str(Path(root)/'output.json'),'--repeats','3']
        if no_email:argv+=['--no-email']
        with patch.object(common,'environment',return_value={'torch':torch.__version__}), \
             patch('src.proj.util.web.task_mail.TaskResultMail',return_value=sender):
            return common.run_cli(common.parser_for('test','test'),execute,'test',argv)

    def test_disabled_machine_keeps_results_without_mail_errors(self):
        with tempfile.TemporaryDirectory() as root, patch('src.proj.env.MACHINE.emailable', False):
            sender = Mock(side_effect=AssertionError('must not send'))
            report = self.invoke(lambda args, report: None, root, sender)
            self.assertEqual(report['status'], 'success')
            self.assertTrue(Path(root, 'output.json').exists())
            self.assertFalse(Path(root, 'output.mail').exists())
            sender.assert_not_called()

    def test_disabled_machine_preserves_original_validation_error(self):
        with tempfile.TemporaryDirectory() as root, patch('src.proj.env.MACHINE.emailable', False):
            sender = Mock(side_effect=AssertionError('must not send'))
            def execute(args, report):
                raise RuntimeError('Candidate validation failed')
            with self.assertRaisesRegex(RuntimeError, 'Candidate validation failed'):
                self.invoke(execute, root, sender)
            sender.assert_not_called()

    def test_success_mail_after_log_closed(self):
        with tempfile.TemporaryDirectory() as root:
            def execute(args,report):
                report['rows'].append(dict(case='one',**common.summarize([1.,2.,3.],2.)))
                print('last experiment marker')
            def send(*args,**kwargs):
                attachments=kwargs['attachments']
                self.assertEqual({p.suffix for p in attachments},{'.json','.csv','.log'})
                self.assertIn('Complete: success',Path(root,'output.log').read_text())
                return True
            sender=Mock(side_effect=send)
            report=self.invoke(execute,root,sender)
            self.assertEqual(report['status'],'success')
            sender.assert_called_once()
            self.assertEqual(json.loads(Path(root,'output.json').read_text())['rows'][0]['median_ms'],2.)

    def test_failure_preserves_rows_mails_and_raises(self):
        with tempfile.TemporaryDirectory() as root:
            def execute(args,report):
                report['rows'].append({'case':'completed'})
                raise ValueError('failure sentinel')
            sender=Mock(return_value=True)
            with self.assertRaisesRegex(ValueError,'failure sentinel'):self.invoke(execute,root,sender)
            saved=json.loads(Path(root,'output.json').read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(len(saved['rows']),1)
            self.assertTrue(Path(root,'output.csv').exists())
            self.assertIn('failure sentinel',Path(root,'output.log').read_text())
            sender.assert_called_once()

    def test_delivery_failure_and_no_mail(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'retry only delivery'):
                self.invoke(lambda args,report:None,root,Mock(return_value=False))
            self.assertEqual(json.loads(Path(root,'output.json').read_text())['status'],'success')
        with tempfile.TemporaryDirectory() as root:
            sender=Mock()
            self.invoke(lambda args,report:None,root,sender,no_email=True)
            sender.assert_not_called()

    def test_original_error_survives_mail_failure(self):
        with tempfile.TemporaryDirectory() as root:
            def fail(args,report):raise ValueError('original calculation failure')
            with self.assertRaisesRegex(ValueError,'original calculation failure'):
                self.invoke(fail,root,Mock(return_value=False))
            self.assertEqual(json.loads(Path(root,'output.json').read_text())['status'],'failed')

    def test_cuda_request_never_falls_back(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'CUDA requested but unavailable'):
                    hidden.main(['--device','cuda','--no-email','--output',str(Path(root)/'cuda.json')])
            self.assertEqual(json.loads(Path(root,'cuda.json').read_text())['status'],'failed')

    def test_timing_samples_and_overwrite_protection(self):
        samples,loops=common.timed_variants({'a':lambda:None,'b':lambda:None},3)
        self.assertGreater(loops,0)
        self.assertEqual([len(v) for v in samples.values()],[3,3])
        with tempfile.TemporaryDirectory() as root:
            Path(root,'output.json').write_text('do not overwrite')
            with self.assertRaises(SystemExit):self.invoke(lambda a,r:None,root,Mock())
            self.assertEqual(Path(root,'output.json').read_text(),'do not overwrite')
