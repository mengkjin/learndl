# ResNet-GRU 日内聚合重训

两个生产目标配置固定使用 linear + 15m histnorm。实验配置也固定开启 histnorm，不做开关参数扫描。

- resnet_gru_new_15m_pool：model_num 0–4 分别为 flatten、mean、linear、attention、单层 depthwise conv。
- resnet_gru_new_15m_pool_conv2：model_num 0 为两层 depthwise conv。

两份配置种子均为 0，其他训练设置沿用 resnet_gru_new_15m；使用独立模型名称，首次运行 resume=0。后续种子 1、2 的复核应另存配置并更换模型名称，避免覆盖结果。

## 先重算历史统计

在有完整 fit 数据的机器、仓库根目录执行：

```python
from src.data.util import DataBlock
from src.data.preprocess.hist_norm import PreProHistNorm
for key in ('day', '15m', '30mcont'):
    PreProHistNorm.calculate(DataBlock.load_preprocess(key, frame='fit'), key,
                            start=20070101, end=20161231)
```

统计只能使用 2016 年末及之前的数据，预测复用 fit 统计。15m 使用 [16,F]，在每个跨日位置广播。30mcont 换手率统一除以最后一天总量，不再除以整个窗口总量；末端只将价格置零。现存 30mcont 模型的输入行为随修复改变，需要重新训练。

## 启动实验

在同一环境逐份运行（本次代码交付未执行）：

```python
from src.res.model.model_module.application.trainer import ModelTrainer
ModelTrainer.schedule(schedule_name='resnet_gru_new_15m_pool', resume=0)
ModelTrainer.schedule(schedule_name='resnet_gru_new_15m_pool_conv2', resume=0)
```

比较样本外 IC/RankIC、年度稳定性、训练/验证差距、收敛速度与参数量。linear 是共享的时间投影，允许正负时段对比；mean 更强约束；attention 为按样本动态加权；depthwise conv 保持轻量的通道独立时间压缩。不能仅凭单次最佳结果认定改善。
