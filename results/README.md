# 结果目录

新接收产物按研究类型、来源电脑、实验、批次组织：

```text
<研究类型>/<电脑>__<实验名称>/<UTC时间>__<UUID>/
  results/                 # 完整的已登记分析结果
  tensorboard/<名称>.zip   # 可选
  optuna/<名称>.zip        # 可选
  source.json              # 来源、状态、运行 ID 和校验清单
```

研究类型包括 model_training、model_compare、factor_sequence、rl 和脚本声明的自定义类型。模型结果来自该模型自身的 results 文件夹，不是本目录的整体复制。实验面板从 source.json 发现同步实验，并继续支持旧实验 ZIP。

- `model_experiments/{tensorboard,optuna}/`：旧实验 ZIP，保留面板兼容。
- `reports/`：旧版接收结果和历史迁移报告；不再用于新批次分类。
- `rl_experimental`、`factor_sequence`、`nn_training_review` 等既有运行目录保留，接收器只操作自己的新批次，不清理本机工作目录。
- `data_repairs` 等其他目录用途不变，autofix 不投递。

共享区 `$HOME/workspace/Syncthing/artifact_exchange` 只消费完成发布的 transfer.zip。全部登记载荷成功归档并持久化回执后删除共享包，服务器原件保留。普通手工文件请先用 CLI 发布。

历史迁移链接保持有效；model_compare 的既有兼容链接也可用于访问新批次，其实际存储位置仍在旧链接目标下。

完整安装、配置、脚本接口与恢复步骤见 [运行说明](../runs/artifact_exchange/README.md)。
