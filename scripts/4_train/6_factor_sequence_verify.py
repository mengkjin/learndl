#!/usr/bin/env python3
# description: Factor sequence offline/build/audit/walk-forward smoke/compare with result mail
# email: false
# mode: shell
# parameters:
#   run_id:
#     type: str
#     default: null
#     desc: Unique run ID; empty generates one automatically
#   test_start:
#     type: int
#     default: 20170101
#     desc: First requested out-of-sample date
#   test_end:
#     type: int
#     default: null
#     desc: Last requested date; empty uses latest mature label
#   research:
#     type: bool
#     default: false
#     desc: Three seeds and up to 30 epochs
#   ablations:
#     type: bool
#     default: false
#     desc: Include three additional ablations
#   data_config:
#     type: str
#     default: configs/experience/model/factor_sequence/data.json
#     desc: Frozen data configuration
#   recipient:
#     type: str
#     default: null
#     desc: Email recipient; empty uses project default
#   no_email:
#     type: bool
#     default: false
#     desc: Explicitly disable result mail

# Email is owned by this script, including failed-stage reports.
from pathlib import Path
import argparse
from datetime import datetime, timezone
import sys
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default=None)
    parser.add_argument('--test-start',type=int,default=20170101)
    parser.add_argument('--test-end',type=int,help='Default: latest mature label discovered on this server')
    parser.add_argument('--data-config',default='configs/experience/model/factor_sequence/data.json')
    parser.add_argument('--research',action='store_true')
    parser.add_argument('--ablations',action='store_true')
    parser.add_argument('--recipient',help='Omit to use the configured project recipient')
    parser.add_argument('--no-email',action='store_true')
    return parser.parse_args(argv)


def main(run_id=None, test_start=20170101, test_end=None,
         data_config='configs/experience/model/factor_sequence/data.json',
         research=False, ablations=False, recipient=None, no_email=False):
    from src.proj import PATH
    from src.data.preprocess.factor_sequence.server_suite import run_suite, send_result
    from src.data.preprocess.factor_sequence.bundle import write_json
    run_id=run_id or 'fs_server_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6]
    config=Path(data_config)
    if not config.is_absolute():
        config=PATH.main/config
    folder,report=run_suite(PATH.main,PATH.result/'factor_sequence',run_id,config,
        test_start,test_end,research,ablations)
    if no_email:
        mail={'status':'skipped','reason':'--no-email'}
        write_json(folder/'mail_status.json',mail)
    else:
        mail=send_result(folder,recipient)
    print(f"Experiment: {report['status']}; mail: {mail['status']}; results: {folder}",flush=True)
    if report['status']!='passed' or mail['status']=='failed':
        raise RuntimeError(f'Factor sequence verification failed; inspect {folder}')
    return 0


if __name__=='__main__':
    raise SystemExit(main(**vars(parse_args())))
