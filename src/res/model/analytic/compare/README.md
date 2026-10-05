# 模型对比

独立读取训练报告的 IC 与 Top 组合曲线，重新统计指定窗口并计算互补性。
不重新运行因子测试、组合回测或训练。默认同时生成 `comparison.xlsx` 和 `comparison.pdf`。
同时保存 `compare_parameters.json`，记录完整配置、模型来源与 hidden 副本编号、
各模板实际日期范围与样本数、统计口径、导出时间和目录，方便核对与复用参数。

## CLI 菜单

CLI 菜单入口：运行 `python cli.py`，选择 **Research Operations → Compare Models**。
先多选至少两个已有 `results/detailed_alpha_data.xlsx` 的模型，再确认是否使用默认参数；
选择自定义时可调整日期、分期、模板、基准/策略、相关方法、pred/hidden 开关及采样上限等。
确认参数后直接计算、展示重点结果并导出 Excel/PDF；默认保存到 `results/model_compare/<时间戳>/`。
在选择或参数步骤退出不会启动计算。

## Python

```python
from src.res.model.analytic.compare import (
    CompareConfig, CompareModelSpec, CompareResult, compare_models,
)

# 显式 Excel 不会根据文件名猜测模型目录。
result = compare_models(
    [CompareModelSpec('results/a.xlsx', name='A'),
     CompareModelSpec('results/b.xlsx', name='B')],
    config=CompareConfig(start=20200101, end=20260807),
    output_dir='results/model_compare/example',
)

# 使用真实训练目录；报告路径默认是目录下的 results/detailed_alpha_data.xlsx。
result = compare_models(
    [CompareModelSpec('models/nn/gru@model_a', name='A', hidden_model_num=0),
     CompareModelSpec('models/nn/gru@model_b', name='B', hidden_model_num=1)],
    config=CompareConfig(
        analyze_pred=True, analyze_hidden=True,
        sample_num_corr=20, sample_num_hidden=20,
        periods=('all', 'year', 'quarter', 'month', 'recent_year'),
        custom_periods={'event': (20240101, 20240331)},
    ),
    display=False,
)
result.display()
print(result.summary, result.output_paths)
result.export('results/model_compare/example_copy')
```

将历史 Excel 与推理目录关联时，显式设置
`CompareModelSpec('historical.xlsx', inference_dir='models/nn/gru@model_a')`。
调用者需确保这些归档确实对应该历史报告；文件名本身不能证明版本一致。

默认配置：`best` 子模型、`market` IC、`t50@perf_curve` 中的
`Top_50/univ/lag0/topN=50`。不匹配时报出可用组合，不自动选择另一策略。
使用 `ic_benchmark`、`top_sheet`、`top_strategy`、`top_benchmark`、`top_suffix`、`top_n`
切换对象。`templates` 支持 `ic`、`top`、`complementarity`；互补性依赖 IC 与 Top 数据。

## 命令行

```sh
# 无输入时，通过项目 CLI 多选已有训练报告的模型。
python scripts/5_test/2_compare_models.py

python scripts/5_test/2_compare_models.py results/a.xlsx results/b.xlsx \
  --start 20200101 --periods all year recent_year --no-display

python scripts/5_test/2_compare_models.py --specs compare.json --pred --hidden \
  --sample-num-corr 20 --sample-num-hidden 10 --custom-period event:20240101:20240331
```

`compare.json` 为 `CompareModelSpec` 字典列表：

```json
[
  {"model": "results/a.xlsx", "name": "A", "inference_dir": "models/nn/gru@model_a", "hidden_model_num": 0},
  {"model": "results/b.xlsx", "name": "B", "inference_dir": "models/nn/gru@model_b", "hidden_model_num": 1}
]
```

## 口径与缺失数据

- IC 与 Top 分别按共同日期对齐，各自保留实际起止日和观察数；不将 IC 抽样频率当成日频。
- IC 滚动窗口默认 20 个观察值，累计 IC 和窗口内统计全部重算。
- `pf/bm` 复利反推每日收益，`excess` 差分反推每日超额；在截取窗口前完成反推。
- 总收益采用复利，累计每日超额采用加法；两者不混用。年化超额使用
  `prod(1 + daily_excess) ** (365 / calendar_days) - 1`，TE 使用总体标准差与对应年化因子，IR 为年化超额 / TE。
- 回撤包含窗口起点零。缺失交易日之后的跨日差分不当作单日收益；有缺口的期间，
  日频年化、TE、IR、回撤不可用，但起点基线已知时仍可从累计端点确定区间收益及超额和。
  图表在缺口处断线并保留已知端点；不填零、不插值。
- 日历默认使用项目 `CALENDAR`，不受本机“当前日期”截断；离线可传入完整、权威的
  `trade_dates`。不得只传源文件日期以掩盖交易日缺口。
- IC/收益相关性采用同日有限值配对，至少三个观察值。常量序列、无数据返回 NA。
  Excel 各相关性表同时列出每个矩阵对应的有效配对数。
- pred/hidden 各自在可用模型的共同候选日期上等距离采样，日期上限独立；失败不补采。
  无对应推理目录的 Excel 可完成基础分析，可选输出分析则记录不可用原因。
- pred 优先读训练快照，保持 `which_output` 与副本均值口径。
  hidden 只使用指定副本，不平均不同训练副本的神经元。
- 补推理选择每个副本严格早于采样日期的 checkpoint；同日 pred/hidden 复用前向计算。
  新表示写入现有 `snapshot/pred_values`、`snapshot/hidden_values` 缓存，不改写训练预测文件。
- hidden 的维度相关性保留日期、模型对及 checkpoint，按日期分表保存。
  跨日期只汇总中心化线性 CKA，公式为
  `||X.T @ Y||_F**2 / (||X.T @ X||_F * ||Y.T @ Y||_F)`，不逐维标准化。
  CKA 使用各模型对完整有限的共同股票，可比较不同维数；退化表示返回 NA。
- 逐日 pred 相关性、CKA 的汇总为日期等权算术均值，另附标准差和成功日期数。

## 输出与验证

`CompareResult` 暴露 `summary`、`tables`、`correlations`、`figures`、`samples`、
`diagnostics`、`metadata`、`highlights` 和 `output_paths`。
Excel 中的 `Sheet_index` 保存原始表名与分表对应关系。每个相关性指标的所有分期
集中在一张表，PDF 只展示全区间主矩阵；完整分期和 hidden 维度明细保留在 Excel。
绘图复用项目的 `new_figure`、`plot_table`、`set_xaxis`、`set_yaxis` 和 seaborn 主题，
采用等宽字体、宽幅画布、蓝色表头与浅蓝条纹。同一模型跨图保持相同颜色，
绿红调色板接近白色的中间色加深为灰色，以保证白底上的辨识度。

不指定输出目录时创建带微秒时间戳的新目录；指定目录时覆盖该目录的同名对比报告。
导出同步完成，写文件异常直接抛出。

```sh
python -m unittest discover -s tests -p test_model_compare.py -v
```
