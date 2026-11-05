"""Exercise resume arguments through the real schedule script's filtering wrapper."""
from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.pkgs.model import ModelAPI


class ScheduleScriptResumeTest(unittest.TestCase):
    def test_explicit_resume_and_directory_survive_script_filter(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/4_train/2_schedule_model.py'
        main = runpy.run_path(str(script))['main']
        # Keep the real ScriptTool argument filter, bypass process admission,
        # locks, recording and notifications; no production training is run.
        with patch('src.proj.util.functional.parallel.is_main_process', return_value=False), \
                patch.object(ModelAPI, 'schedule_model') as train:
            for resume in (True, False):
                with self.subTest(resume=resume):
                    main(schedule_name='gru_gru_15m', short_test=False, resume=resume,
                         base_path='/tmp/existing-model', resume_selection='latest', email=True)
                    train.assert_called_with(
                        'gru_gru_15m', False, int(resume), start=None, end=None,
                        resume_selection='latest', base_path=Path('/tmp/existing-model'))


if __name__ == '__main__':
    unittest.main()
