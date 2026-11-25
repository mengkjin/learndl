"""Production batch-cache contract; all disk operations stay in temporary roots."""

from contextlib import nullcontext
from inspect import unwrap
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd
import torch

from src.proj import PATH
from src.res.model.util.storage.torch_file import (
    TorchFileStorage,
    StoredTorchFileLoader,
)


class TemporaryStorage(TorchFileStorage):
    def __init__(self, memory, directory):
        super().__init__(memory)
        self.directory = Path(directory)

    def real_path(self, key):
        return self.directory / f"{key}.pt"


class TorchFileStorageTests(TestCase):
    def test_live_disk_stores_with_identical_keys_are_isolated(self):
        with tempfile.TemporaryDirectory() as root, patch.object(PATH, 'minibatch', Path(root)):
            first, second = TorchFileStorage(), TorchFileStorage()
            self.addCleanup(first.close_disk)
            self.addCleanup(second.close_disk)
            first.save(torch.tensor([1]), 'train.0', 'fit')
            second.save(torch.tensor([2]), 'train.0', 'fit')
            first_path = first.real_path('train.0')
            second_path = second.real_path('train.0')
            self.assertNotEqual(first_path.parent, second_path.parent)
            first.del_group('fit')
            first.close_disk()
            self.assertFalse(first_path.parent.exists())
            torch.testing.assert_close(second.load('train.0'), torch.tensor([2]))

    def test_restart_reaps_killed_process_cache_only(self):
        script = """
import sys
from pathlib import Path
from src.proj import PATH
from src.res.model.util.storage.torch_file import TorchFileStorage

PATH.minibatch = Path(sys.argv[1])
storage = TorchFileStorage()
storage.real_path('train.0').write_bytes(b'partial data')
sys.stdin.read()
"""
        with tempfile.TemporaryDirectory() as root, patch.object(PATH, 'minibatch', Path(root)):
            live = TorchFileStorage()
            self.addCleanup(live.close_disk)
            live_path = live.real_path('train.0')
            legacy_path = Path(root) / 'train.0.pt'
            legacy_path.touch()
            child = subprocess.Popen([sys.executable, '-c', script, root], stdin=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 20
                paths = []
                while time.monotonic() < deadline and child.poll() is None:
                    paths = [path for path in (Path(root) / 'runs').glob('batch-*/train.0.pt')
                             if path != live_path]
                    if paths:
                        break
                    time.sleep(.05)
                self.assertEqual(len(paths), 1)
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=10)
            restarted = TorchFileStorage()
            self.addCleanup(restarted.close_disk)
            restarted.real_path('train.0')
            self.assertFalse(paths[0].parent.exists())
            self.assertTrue(live_path.parent.exists())
            self.assertTrue(legacy_path.exists())

    def test_trainer_releases_fit_batches_before_test_and_cleans_interruption(self):
        from src.res.model.util.data.data_module import DataModule
        from src.res.model.util.trainer.base_trainer import BaseTrainer

        with (tempfile.TemporaryDirectory() as root, patch.object(PATH, 'minibatch', Path(root)),
              patch('src.proj.util.memory_tracker.ProcessMemoryTracker', return_value=nullcontext(Mock())),
              patch('src.automation.artifact_exchange.research.model_run', return_value=nullcontext())):
            config = SimpleNamespace(mem_storage=False, model_name='test')
            data = DataModule(config)  # type: ignore[arg-type]
            trainer = Mock(data=data, config=config, base_path=SimpleNamespace(base=Path(root) / 'model'),
                           queue_of_stages=['fit', 'test'])
            trainer.model_phase_log.side_effect = lambda _: nullcontext()
            directory = data.storage.real_path('train.0').parent

            def fit():
                data.storage.save(torch.tensor([1]), 'train.0', 'fit')

            def evaluate():
                self.assertFalse((directory / 'train.0.pt').exists())
                data.storage.save(torch.tensor([2]), 'test.0', 'test')
                raise KeyboardInterrupt()

            trainer.stage_fit.side_effect = fit
            trainer.stage_test.side_effect = evaluate
            with self.assertRaises(KeyboardInterrupt):
                unwrap(BaseTrainer.main_process)(trainer)
            self.assertFalse(directory.exists())

    def test_overwrite_move_groups_and_delete(self):
        for memory in (True, False):
            with self.subTest(memory=memory), tempfile.TemporaryDirectory() as root:
                store = TemporaryStorage(memory, root)
                store.save(torch.tensor([1.0]), "a", "g0")
                store.save(torch.tensor([2.0]), "b", "g1")
                store.save(torch.tensor([3.0]), "c", "g0")
                store.save(torch.tensor([4.0]), "a", "g0")
                store.save(torch.tensor([5.0]), "b", "g0")
                self.assertEqual(store.keys, ["a", "b", "c"])
                self.assertEqual(store.records["key"].tolist(), ["c", "a", "b"])
                self.assertEqual(store.records.index.tolist(), [0, 0, 0])
                store.del_group("g1")  # Overwritten b must not remain in old group.
                self.assertTrue(store.exists("b"))
                torch.testing.assert_close(store.load("b"), torch.tensor([5.0]))
                store.save(torch.tensor([6.0]), "c", "g2")
                store.del_group("g0")
                self.assertEqual(store.keys, ["c"])
                self.assertFalse(store.exists("a"))
                store.del_group("g0")
                store.del_group("missing")
                with self.assertRaises(KeyError):
                    store.del_one("missing")
                self.assertEqual(store.keys, ["c"])
                store.del_one("c")
                store.del_all()
                self.assertTrue(store.records.empty)
                store.save(torch.tensor([7.0]), "d")
                store.del_all()
                store.del_all()
                self.assertEqual(store.keys, [])
                self.assertEqual(store._groups, {})
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_dataframe_is_independent_query_snapshot(self):
        for memory in (True, False):
            with self.subTest(memory=memory), tempfile.TemporaryDirectory() as root:
                store = TemporaryStorage(memory, root)
                expected_empty = pd.DataFrame(
                    columns=pd.Index(["key", "path", "group"]), dtype=str
                )
                pd.testing.assert_frame_equal(store.records, expected_empty)
                store.save(torch.ones(2), "a", "g")
                store.save(torch.zeros(2), "b", "other")
                path = "a" if memory else str(Path(root) / "a.pt")
                expected = pd.DataFrame(
                    [{"key": "a", "path": path, "group": "g"}], index=[0]
                )
                pd.testing.assert_frame_equal(
                    store.records.query("group == 'g'"), expected
                )
                snapshot = store.records
                snapshot.loc[:, "group"] = "changed"
                self.assertEqual(store.records["group"].tolist(), ["g", "other"])
                store.save(torch.ones(2), "a", "new")
                self.assertEqual(snapshot["key"].tolist(), ["a", "b"])
                self.assertEqual(store.records["key"].tolist(), ["b", "a"])

    def test_static_loader_and_missing_load(self):
        for memory in (True, False):
            with self.subTest(memory=memory), tempfile.TemporaryDirectory() as root:
                store = TemporaryStorage(memory, root)
                for i in range(3):
                    store.save(torch.tensor([i]), str(i), "fit")
                loader = StoredTorchFileLoader(store, ["2", "0", "1"])
                self.assertEqual(len(loader), 3)
                self.assertEqual([int(x.item()) for x in loader], [2, 0, 1])
                self.assertEqual(loader[1].item(), 0)
                if memory:
                    with self.assertRaises(KeyError):
                        store.load("missing")
                else:
                    self.assertIsNone(store.load("missing"))
                    store.real_path("1").unlink()
                    self.assertIsNone(store.load("1"))
                store.del_all()

    def test_mutation_does_not_materialize_dataframe(self):
        with tempfile.TemporaryDirectory() as root:
            store = TemporaryStorage(True, root)
            with patch(
                "src.res.model.util.storage.torch_file.pd.DataFrame",
                side_effect=AssertionError("hot-path table"),
            ):
                for i in range(480):
                    store.save(None, str(i), f"g{i % 2}")
                store.save(None, "0", "new")
                store.del_group("g0")
                store.del_one("0")
                store.del_all()
            self.assertTrue(store.records.empty)

    def test_failed_payload_operations_preserve_index(self):
        with tempfile.TemporaryDirectory() as root:
            store = TemporaryStorage(False, root)
            store.save(torch.ones(1), "a", "original")
            before = store.records
            with patch("torch.save", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    store.save(torch.zeros(1), "a", "new")
            pd.testing.assert_frame_equal(store.records, before)
            with patch.object(Path, "unlink", side_effect=OSError("unlink failed")):
                with self.assertRaises(OSError):
                    store.del_one("a")
            pd.testing.assert_frame_equal(store.records, before)
            self.assertTrue(store.exists("a"))
            store.del_group("original")
            self.assertTrue(store.records.empty)
