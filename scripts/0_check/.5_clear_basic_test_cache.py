#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2026-07-31
# description: Clear BasicTestResult date-level cache
# content: |
#   Delete all models/**/snapshot/basic_test/test_by_date.* under PATH.model
#   (nn / boost / factor / st / ...) so the next resume/test rebuilds
#   BasicTestResult metrics. Does not touch pred_recorder or detailed_alpha.
# email: True
# mode: shell
# parameters:
#   dry_run:
#       type: [True, False]
#       desc: if True, only list matched files without deleting
#       required: False
#       default: True

from __future__ import annotations

from src.proj import Logger , PATH , DB
from src.proj.util.script import ScriptTool


@ScriptTool('clear_basic_test_cache')
def main(dry_run: bool = False, **kwargs):
    """
    Invalidate BasicTestResult caches under the entire models tree.

    After deletion, run resume_testing / test_model so metrics are rebuilt.
    """
    dry_run = bool(dry_run)
    root = PATH.model
    if not root.exists():
        Logger.skipping(f'Model root not found: {root}')
        return

    # Match any DF suffix under any model subfolder (nn/boost/factor/st/...).
    pattern = f'**/snapshot/basic_test/test_by_date.{DB.DF_SUFFIX}'
    matched = sorted(root.glob(pattern))
    Logger.note(f'Scanning {PATH.relative(root)} ({len(matched)} matches for {pattern})')

    deleted = 0
    for path in matched:
        Logger.alert1(f'{"Would delete" if dry_run else "Deleting"}: {PATH.relative(path)}')
        if not dry_run:
            path.unlink(missing_ok=True)
            deleted += 1

    Logger.note(
        f'BasicTest cache clear done: matched={len(matched)}, '
        f'deleted={deleted if not dry_run else 0}, dry_run={dry_run}'
    )
    if matched and dry_run:
        Logger.note('Re-run with dry_run=False to delete, then resume_testing / test_model to rebuild.')


if __name__ == '__main__':
    main()
