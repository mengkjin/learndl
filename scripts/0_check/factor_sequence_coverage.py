#!/usr/bin/env python3
# description: Read-only factor coverage, slim context and risk audit since 2011; send report mail
# email: false
# mode: shell
# parameters:
#   start:
#     type: int
#     default: 20110101
#     desc: First reporting date (warm-up is loaded separately)
#   end:
#     type: int
#     default: null
#     desc: Last confirmed project date by default
#   no_email:
#     type: bool
#     default: false
#     desc: Disable mail and external artifact delivery
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.proj.util.script import ScriptTool


@ScriptTool("factor_sequence_coverage", markdown_catcher=True)
def main(start=20110101, end=None, no_email=False):
    from src.proj import Logger
    from src.proj import PATH
    from src.data.preprocess.factor_sequence.audit_runner import (
        run_server,
        deliver,
        new_run_id,
    )
    from src.data.preprocess.factor_sequence.context_diagnostics import MODES

    roots = {m: PATH.model_nn / f"factorseq_residual@factorseq_v2_{m}@2" for m in MODES}
    folder = PATH.result / "factor_sequence" / new_run_id()
    report = run_server(folder, start, end, roots)
    mail = deliver(folder) if not no_email else {"status": "skipped"}
    Logger.stdout(f"Audit: {report['status']}; mail: {mail['status']}; {folder}", flush=True)
    if report["status"] not in ("completed",):
        raise RuntimeError(f"Audit incomplete; partial results retained: {folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=20110101)
    p.add_argument("--end", type=int)

    def boolean(value):
        if str(value).lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError("Expected true or false")
        return str(value).lower() == "true"

    p.add_argument(
        "--no-email", "--no_email", type=boolean, nargs="?", const=True, default=False
    )
    p.add_argument("--task_id", help=argparse.SUPPRESS)
    p.add_argument("--source", help=argparse.SUPPRESS)
    p.add_argument("--email", type=boolean, help=argparse.SUPPRESS)
    main(**{k: v for k, v in vars(p.parse_args()).items() if v is not None})
