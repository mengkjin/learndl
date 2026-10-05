"""Training mail names share the html attachment's model suffix."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.res.model.callback.test.detail import DetailedAlphaAnalysis

attachment_name = DetailedAlphaAnalysis.training_report_attachment_name


class TrainingAttachmentNameTest(unittest.TestCase):
    def test_data_and_plot_use_the_html_suffix_from_the_model_name(self) -> None:
        html = 'train_schedule_model_gru_day_lgbm_at_20261004233100.html'
        self.assertEqual(
            attachment_name(Path('detailed_alpha_data.xlsx'), 'gru_day_lgbm', html),
            'detailed_alpha_data_gru_day_lgbm_at_20261004233100.xlsx',
        )
        self.assertEqual(
            attachment_name(Path('detailed_alpha_plot.pdf'), 'gru_day_lgbm', html),
            'detailed_alpha_plot_gru_day_lgbm_at_20261004233100.pdf',
        )

    def test_suffix_keeps_the_html_spelling(self) -> None:
        html = 'Train_Schedule_Model_Gru_Day_Lgbm_at_20261004233100.html'
        self.assertEqual(
            attachment_name(Path('detailed_alpha_data.xlsx'), 'gru_day_lgbm', html),
            'detailed_alpha_data_Gru_Day_Lgbm_at_20261004233100.xlsx',
        )

    def test_model_name_absent_from_html_is_appended_alone(self) -> None:
        self.assertEqual(
            attachment_name(
                Path('detailed_alpha_data.xlsx'), 'gru_day_lgbm', 'fitting_testing_output.html',
            ),
            'detailed_alpha_data_gru_day_lgbm.xlsx',
        )
        self.assertEqual(
            attachment_name(Path('detailed_alpha_plot.pdf'), 'gru_day_lgbm', None),
            'detailed_alpha_plot_gru_day_lgbm.pdf',
        )
