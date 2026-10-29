"""One-shot request and isolated four-process evidence regression tests."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.api.task_monitor.scheduling import nn_metrics_ab as queue
from src.res.model.diagnostics import nn_metrics_ab as runner


class NNMetricsABTest(TestCase):
    def test_idempotent_and_consumed_request(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(queue, 'requests_dir', return_value=Path(tmp)):
            path = queue.submit('test_01')
            self.assertEqual(path, queue.submit('test_01'))
            rows = []
            store = SimpleNamespace(owned=lambda owner: rows)
            request = queue.observe(store)
            rows.append(dict(request, phase='error'))
            self.assertIsNone(queue.observe(store))
            rows[0]['phase'] = 'complete'
            self.assertIsNone(queue.observe(store))
            rows[0]['phase'] = 'deferred'
            self.assertEqual(queue.observe(store), request)
            with self.assertRaises(ValueError):
                queue.submit('../escape')

    def test_four_separate_processes_failure_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('configs/model','src/res/model/util/metric','src/res/model/callback/monitor'):
                (root/folder).mkdir(parents=True)
            (root/'src/res/model/callback/monitor/summary.py').write_text('# diagnostic')
            calls=[]
            def launch(args, **kwargs):
                if '--single' not in args:
                    return SimpleNamespace(returncode=0)
                calls.append((args,kwargs))
                out=Path(args[args.index('--single')+1]); name=args[args.index('--name')+1]
                fast=bool(int(args[-1])); code=1 if len(calls)==2 else 0
                model=root/'models'/name
                (model/'archive').mkdir(parents=True)
                (model/'archive/weights.txt').write_text('evidence')
                runner.save(out/'run.json',dict(name=name,fastpath=fast,status='failed' if code else 'success',model_path=str(model)))
                return SimpleNamespace(returncode=code)
            with patch.object(runner.platform,'platform',return_value='test'), patch.object(runner,'probe',return_value={}), patch.object(runner.subprocess,'run',side_effect=launch):
                result=runner.run_suite(root/'evidence',root,'test_01')
            self.assertEqual([x[0][-1] for x in calls],['0','1','1','0'])
            self.assertTrue(all(x[1]['env']['PYTHONHASHSEED']=='20261029' for x in calls))
            self.assertEqual(result['status'],'failed')
            self.assertEqual(len(result['runs']),4)
            self.assertTrue((root/'evidence/fast_1/model/archive/weights.txt').exists())
            self.assertEqual(json.loads((root/'evidence/suite.json').read_text())['status'],'failed')

    def test_dispatch_honors_busy_and_one_shot_priority(self):
        from contextlib import nullcontext
        from src.api.task_monitor.scheduling import idle_worklist as idle, rl_experiment as rl
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp, patch.object(queue,'requests_dir',return_value=Path(tmp)):
            queue.submit('test_dispatch')
            rows=[]
            store=SimpleNamespace(owned=lambda owner: rows, put=lambda row: rows.append(row))
            with patch.object(idle,'queue_lock',return_value=nullcontext()), \
                 patch.object(idle,'worklist_schedules',return_value=({}, {}, [])), \
                 patch.object(rl,'observe',return_value=None), patch.object(rl,'status',return_value={}), \
                 patch.object(rl,'waiting'), patch.object(idle,'idle_reason',return_value='GPU busy') as busy:
                self.assertEqual(idle.dispatch({'enabled':True},store)['idle_worklist'],'GPU busy')
                self.assertEqual(rows,[])
                busy.return_value=None
                self.assertIn('queued',idle.dispatch({'enabled':True},store))
                self.assertEqual(rows[0]['task_kind'],'nn_metrics_ab')
                self.assertEqual(idle.dispatch({'enabled':True},store)['idle_worklist'],'queued or active')
                rows[0]['phase']='error'
                self.assertEqual(idle.dispatch({'enabled':True},store)['idle_worklist'],'no eligible schedules')

    def test_shared_occupancy_does_not_deadlock_child_fit(self):
        import subprocess
        import sys
        from src.proj.util.script import FitLock, FitLockNN
        with tempfile.TemporaryDirectory() as tmp, patch.object(FitLock,'LOCK_DIR',Path(tmp)), patch.object(FitLockNN,'LOCK_DIR',Path(tmp)):
            with FitLock.guard():
                self.assertTrue(FitLock.is_held())
                code = "import sys,portalocker; a=open(sys.argv[1],'a+'); b=open(sys.argv[2],'a+'); portalocker.lock(a,portalocker.LOCK_SH|portalocker.LOCK_NB); portalocker.lock(b,portalocker.LOCK_EX|portalocker.LOCK_NB)"
                subprocess.run([sys.executable,'-c',code,str(FitLock.lock_path()),str(FitLockNN.lock_path())],check=True,timeout=10)
