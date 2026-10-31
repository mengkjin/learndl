#!/usr/bin/env python3
# description: Factor sequence offline/build/audit/walk-forward smoke/compare with result mail
# email: false
# mode: shell
# Email is owned by this script, including failed-stage reports (no duplicate task email).
from pathlib import Path
import argparse
from datetime import datetime, timezone
import sys
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default=None)
    parser.add_argument('--test-start',type=int,default=20170101)
    parser.add_argument('--test-end',type=int,help='Default: latest mature label discovered on this server')
    parser.add_argument('--data-config',default='configs/experience/model/factor_sequence/data.json')
    parser.add_argument('--research',action='store_true')
    parser.add_argument('--ablations',action='store_true')
    parser.add_argument('--recipient',help='Omit to use the configured project recipient')
    parser.add_argument('--no-email',action='store_true')
    args=parser.parse_args()
    from src.proj import PATH
    from src.data.preprocess.factor_sequence.server_suite import run_suite, send_result
    from src.data.preprocess.factor_sequence.bundle import write_json
    run_id=args.run_id or 'fs_server_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6]
    config=Path(args.data_config)
    if not config.is_absolute():
        config=PATH.main/config
    folder,report=run_suite(PATH.main,PATH.result/'factor_sequence',run_id,config,
        args.test_start,args.test_end,args.research,args.ablations)
    if args.no_email:
        mail={'status':'skipped','reason':'--no-email'}
        write_json(folder/'mail_status.json',mail)
    else:
        mail=send_result(folder,args.recipient)
    print(f"Experiment: {report['status']}; mail: {mail['status']}; results: {folder}",flush=True)
    return 0 if report['status']=='passed' and mail['status']!='failed' else 1


if __name__=='__main__':
    raise SystemExit(main())
