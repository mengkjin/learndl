#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-06-02
# description: clear catcher logs
# content: clear catcher logs
# email: True
# mode: shell
# parameters:
#   days_ago : 
#       type : int
#       desc : clear catcher logs that are older than days_ago days
#       required : True
#       default : 30

from collections import defaultdict
from datetime import datetime , timedelta
from src.proj import PATH , Logger
from src.proj.util.script import ScriptTool

@ScriptTool('clear_catchers')
def main(days_ago : int = 30 , **kwargs): 
    assert days_ago > 2 , 'days_ago must be greater than 2'
    root = PATH.logs.joinpath('catcher')
    cleared_counts : dict[str, int] = defaultdict(int)
    for sub_catcher in root.iterdir():
        if sub_catcher.name.startswith('.'):
            continue
        assert sub_catcher.is_dir() , f'{sub_catcher} is not a directory'
        for log_file in sub_catcher.rglob('*'):
            if log_file.is_file() and log_file.stat().st_mtime < (datetime.now() - timedelta(days=days_ago)).timestamp():
                cleared_counts[str(log_file.parent.relative_to(root))] += 1
                log_file.unlink()
            if log_file.is_dir() and not list(log_file.glob('*')):
                log_file.rmdir()
                cleared_counts['empty_dir'] += 1

    Logger.stdout_pairs(cleared_counts , title = f'Cleared {sum(cleared_counts.values())} log files in total, details:')

if __name__ == '__main__':
    main()