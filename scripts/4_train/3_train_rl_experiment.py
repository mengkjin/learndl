#!/usr/bin/env python3
# author: jinmeng
# date: 2026-10-09
# description: Train RL Experiment
# content: 强化学习组合实验：准备数据、训练、分析、打包并发送邮件；默认使用 CUDA，参数在 JSON 配置文件中修改。
# email: False
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
#       desc: 是否跳过实验附件邮件
#       default: False

import json
from pathlib import Path

from src.proj.util.script import ScriptTool


@ScriptTool('train_rl_experiment')
def main(
    config_path: str = 'src/res/rl_experimental/example_server_experiment.json',
    recipient: str | None = None,
    no_email: bool = False,
    **kwargs,
):
    from src.res.rl_experimental.experiment import run_experiment

    # ScriptTool sets the working directory to the project root.
    config = Path(config_path).expanduser().resolve()
    status = run_experiment(config, no_email=no_email, recipient=(recipient or '').strip() or None)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not no_email:
        parts = status.get('delivery', {}).get('parts', {})
        if not parts or any(part.get('sent') is not True for part in parts.values()):
            raise RuntimeError(
                '训练结果已保留，但邮件未全部发送成功。请使用 resend-bundle 重发，'
                f"无需重新训练：{status['bundle_metadata']}"
            )
    return status


if __name__ == '__main__':
    main()
