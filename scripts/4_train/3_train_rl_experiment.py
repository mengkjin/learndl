#!/usr/bin/env python3
# author: jinmeng
# date: 2026-10-09
# description: Train RL Experiment
# content: 强化学习组合实验：准备数据、训练、分析、打包并发送邮件；支持 YAML 配置和原 JSON 配置。
# email: True
# mode: shell
# parameters:
#   config_path:
#       type: str
#       desc: 实验 YAML 或 JSON 配置路径，相对项目根目录
#       default: configs/strategy/rl_experiment.yaml
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
from tempfile import TemporaryDirectory

from src.proj import Proj
from src.proj.util.script import ScriptTool


script_tool = ScriptTool('train_rl_experiment')


@script_tool
def main(
    config_path: str = 'configs/strategy/rl_experiment.yaml',
    recipient: str | None = None,
    no_email: bool = False,
    **kwargs,
):
    from src.res.rl_experimental.bundle import bundle_attachments
    from src.res.rl_experimental.experiment import run_experiment
    from src.automation.scheduling.rl_experiment import parse_config

    task = script_tool.autorun_task
    if no_email:
        task.kwargs['email'] = False
    # ScriptTool sets the working directory to the project root.
    config = Path(config_path).expanduser().resolve()
    content = config.read_text()
    launch = parse_config(content, yaml_format=config.suffix.lower() in {'.yaml', '.yml'})
    print(f'[RL/run] Selected experiment mode: {launch["mode"]}', flush=True)
    supplied = launch['experiment']
    if no_email or launch['no_email']:
        task.kwargs['email'] = False
    supplied['recipient'] = (recipient or '').strip() or launch['recipient']
    task.kwargs['email_recipient'] = supplied['recipient']

    def attach_bundle(metadata: Path) -> None:
        if task.email:
            from uuid import uuid4
            from src.proj.util.web.task_mail import TaskResultMail
            task.email_sender = TaskResultMail(metadata.parent/'task_mail'/uuid4().hex)
            for attachment in bundle_attachments(metadata):
                Proj.email_attachments.append(attachment)

    # Register diagnostic bundles before run_experiment re-raises a failure.
    # AutoRunTask sends these alongside HtmlCatcher output after capture closes.
    # Resolve both input formats once; the existing experiment engines retain
    # their JSON interface. The engine persists this config in its result bundle.
    with TemporaryDirectory(prefix='learndl-rl-config-') as temporary:
        if config.suffix.lower() in {'.yaml', '.yml'}:
            supplied['source_configuration'] = dict(path=str(config), content=content)
        resolved = Path(temporary) / 'experiment.json'
        resolved.write_text(json.dumps(supplied, ensure_ascii=False, indent=2))
        if launch['mode'] == 'walk_forward':
            from src.res.rl_experimental.walk_forward import run_walk_forward
            status = run_walk_forward(resolved, stage=supplied.get('stage', 'all'), bundle_ready=attach_bundle)
        else:
            status = run_experiment(resolved, no_email=True, bundle_ready=attach_bundle)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


if __name__ == '__main__':
    main()
