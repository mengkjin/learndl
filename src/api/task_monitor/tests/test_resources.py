import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.task_monitor.resources import sample_resources


class MonitorResourcesTest(unittest.TestCase):
    def setUp(self):
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('src.api.task_monitor.resources.psutil.virtual_memory',
                                      return_value=MagicMock(total=1000, available=400, percent=60)))
        self.stack.enter_context(patch('src.api.task_monitor.resources.psutil.cpu_percent', return_value=0))
        self.stack.enter_context(patch('src.api.task_monitor.resources.psutil.disk_usage',
                                      return_value=MagicMock(used=250, total=1000, percent=25)))
        self.run = self.stack.enter_context(patch('src.api.task_monitor.resources.subprocess.run'))

    def test_multiple_gpus_and_real_zero(self):
        self.run.return_value.stdout = '0, NVIDIA A, 1024, 4096, 0\n1, NVIDIA B, 4096, 8192, 90\n'
        result = sample_resources(Path('/tmp'))
        self.assertEqual(result.memory, (600, 1000, 60))
        self.assertEqual(result.cpu_percent, 0)
        self.assertEqual([gpu.memory_percent for gpu in result.gpus], [25, 50])
        self.assertEqual([gpu.utilization for gpu in result.gpus], [0, 90])
        self.assertEqual(result.errors, ())
        self.run.assert_called_once()
        self.assertEqual(self.run.call_args.kwargs['timeout'], 2)
        self.assertIn('utilization.gpu', self.run.call_args.args[0][1])

    def test_partial_unknown_fields_and_malformed_rows(self):
        self.run.return_value.stdout = '0, A, N/A, 8192, 32\n1, B, 0, 0, [Not Supported]\nbad row\n'
        result = sample_resources(Path('/tmp'))
        self.assertIsNone(result.gpus[0].used_mib)
        self.assertEqual(result.gpus[0].total_mib, 8192)
        self.assertEqual(result.gpus[0].utilization, 32)
        self.assertIsNone(result.gpus[1].memory_percent)
        self.assertIsNone(result.gpus[1].utilization)
        self.assertTrue(result.errors)

    def test_gpu_probe_failures_preserve_host_metrics(self):
        for exc in (FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 2),
                    subprocess.CalledProcessError(1, 'nvidia-smi'), OSError('unsupported')):
            with self.subTest(exc=exc):
                self.run.side_effect = exc
                result = sample_resources(Path('/tmp'))
                self.assertEqual(result.gpus, ())
                self.assertTrue(result.errors)
                self.assertIsNotNone(result.memory)
                self.assertIsNotNone(result.disk)

    def test_no_devices_and_invalid_numbers(self):
        self.run.return_value.stdout = ''
        self.assertTrue(sample_resources(Path('/tmp')).errors)
        self.run.return_value.stdout = '0, A, nan, -1, 101\n'
        gpu = sample_resources(Path('/tmp')).gpus[0]
        self.assertIsNone(gpu.used_mib)
        self.assertIsNone(gpu.total_mib)
        self.assertIsNone(gpu.utilization)

    def test_resource_cache_reuses_sample_within_refresh_interval(self):
        from src.api.task_monitor.launch import _resources
        from src.api.task_monitor.resources import ResourceSnapshot
        _resources.clear()
        self.addCleanup(_resources.clear)
        sample = ResourceSnapshot(100, None, None, None, (), ())
        with patch('src.api.task_monitor.launch.sample_resources', return_value=sample) as probe:
            self.assertEqual(_resources(), sample)
            self.assertEqual(_resources(), sample)
        probe.assert_called_once()
