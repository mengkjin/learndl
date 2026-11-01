import unittest
from src.automation.scheduling.auxiliary import control_notice


class NoticeTests(unittest.TestCase):
    def test_admission_timeout_is_not_corruption(self):
        for owner in ('git', 'autofix:example'):
            detail=f'RuntimeError: {owner}: Maintenance admission timed out after one hour (request 1793492012.8353074)'
            title, body=control_notice(detail)
            self.assertEqual(title,'Maintenance admission delayed')
            self.assertIn('15-minute cooldown',body)
            self.assertNotIn('Autofix is paused',body)
            self.assertIn(detail,body)

    def test_real_control_failure_keeps_warning(self):
        title, body=control_notice('RuntimeError: Autofix ledger is corrupt; restore from backup')
        self.assertEqual(title,'Autofix control failure')
        self.assertIn('restore the existing control state',body)


if __name__=='__main__':
    unittest.main()
