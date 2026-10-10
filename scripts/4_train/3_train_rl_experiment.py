#!/usr/bin/env python3
# author: jinmeng
# date: 2026-10-09
# description: Train RL Experiment
# content: 强化学习组合实验：准备数据、训练、分析、打包并发送邮件；默认使用 CUDA，参数在 JSON 配置文件中修改。
# email: True
# mode: shell
# parameters:
#   config_path:
#       type: str
#       desc: 实验 JSON 配置路径，相对项目根目录
#       default: src/res/rl_experimental/example_server_experiment.json
#       required: True
#   recipient:
#       type: str
#       desc: 收件人，留空使用实验配置或项目默认邮箱
#   no_email:
#       type: [True, False]
#       desc: 是否跳过整个任务邮件（包括 HTML 日志和实验附件）
#       default: False

import json
from pathlib import Path

from src.proj import Proj
from src.proj.util.script import ScriptTool


script_tool = ScriptTool('train_rl_experiment')


@script_tool
def main(
    config_path: str = 'src/res/rl_experimental/example_server_experiment.json',
    recipient: str | None = None,
    no_email: bool = False,
    **kwargs,
):
    from src.res.rl_experimental.bundle import bundle_attachments
    from src.res.rl_experimental.experiment import load_run_config, run_experiment

    task = script_tool.autorun_task
    if no_email:
        task.kwargs['email'] = False

    # ScriptTool sets the working directory to the project root.
    config = Path(config_path).expanduser().resolve()
    task.kwargs['email_recipient'] = (recipient or '').strip() or load_run_config(config).get('recipient')

    def attach_bundle(metadata: Path) -> None:
        if task.email:
            for attachment in bundle_attachments(metadata):
                Proj.email_attachments.append(attachment)

    # Register diagnostic bundles before run_experiment re-raises a failure.
    # AutoRunTask sends these alongside HtmlCatcher output after capture closes.
    status = run_experiment(config, no_email=True, bundle_ready=attach_bundle)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


if __name__ == '__main__':
    main()
