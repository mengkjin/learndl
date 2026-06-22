#! /usr/bin/env python
# author: jinmeng
# date: 2026-05-21
# description: Smoke test parallel stock factor update
# content: |
#   Run up to N stock factors on a single trading day with process pool vs for-loop.
#   Use after DataVendor / DFCollection thread-safety changes to verify jobs complete.
# email: True
# mode: shell
# parameters:
#   date:
#       type: int
#       desc: trading day YYYYMMDD (default = data update date)
#       required: False
#       default: -1
#   n_factors:
#       type: int
#       desc: number of stock factors to run (first N updatable calculators)
#       required: False
#       default: 20
#   compare:
#       type: [forloop, process, both]
#       desc: forloop only, process pool only, or run both and stdout timing
#       required: False
#       default: both
#   overwrite:
#       type: [True, False]
#       desc: force recalc on that day even if factor file exists
#       required: False
#       default: True
#   lookback_td:
#       type: int
#       desc: if too few factors on requested date, walk back trading days (max)
#       required: False
#       default: 10

from __future__ import annotations
from src.proj.util.script import ScriptTool
from src.api.calls.test.parallel_factor_calculation import test_parallel_factor_calculation

@ScriptTool('test_parallel_factors', lock_num=0)
def main(
    date: int | None = None,
    n_factors: int | None = 60,
    compare: str | None = 'both',
    overwrite: bool | None = True,
    lookback_td: int | None = 10,
    **kwargs,
) -> None:
    assert compare == 'both' or compare == 'forloop' or compare == 'process'
    return test_parallel_factor_calculation(date, n_factors, compare, overwrite, lookback_td)

if __name__ == '__main__':
    main()
