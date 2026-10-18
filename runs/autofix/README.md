# Autofix：发布一次，服务器尝试一次

Autofix 用来在代码更新后执行明确声明的数据重算、回补或修复。你在本电脑维护 `tasks.yaml` 并提交 Git，服务器 watchdog 发现请求后交给独立 worker。服务器不修改 Git 中的文件来标记完成，也不需要把执行状态传回本机。

当前清单注册了 `20261014_create_open_vwap_datablocks_01`：先生成六组 open/VWAP 原始标签，再构建 `y_open` / `y_vwap` 的 fit 与 predict DataBlock。其他脚本没有因本功能而自动获得 autofix 准入。

该任务固定从标签日下一交易日起持有 5、10、20 天，从 20050101 回补至执行时已成熟的日期，不重算原有 close 标签。已有同名原始标签和 DataBlock 会先备份到 `data/Interim/autofix_backups/`，其 `report.json` 记录备份映射、完成步骤和失败原因。六组原始标签全部成功后才构建 DataBlock；失败会保留备份和已完成的数据，不自动重试或回滚。当前统一强制开启脚本结果邮件，超时为 48 小时。`dry_run: true` 只预览数据范围，不生成数据或备份。

## 最重要的规则

当前重跑任务为 `20261018_repair_15mraw_vwap_coverage_train_04`，旧 `_inf_train_03` 已停用。服务器 20261018 09:57 的失败日志运行在 `b9c56b3d`，尚未包含 `44438404` 的 VWAP-to-close 规则；它查到 3 个原始 VWAP Inf，均可由非零 amount / 零 volume 复现，随后因要求 `20070101..20261016` 全日期覆盖、实际重建仅有 4074 个日期而停止。那次日志未打印实际起止日及缺口，不能仅凭总数断定具体缺失年份。

新任务先对 fit/predict **两份数据都做原始日期预检**，使用与生产一致的 `trade_ts/15min` 加备用源日期并集。fit 的有效起点取请求区间内实际原始数据首日，允许排除原始行情尚未开始的前置历史；predict 不允许缩短请求窗口。有效起点到固定截止日之间的中间/尾部缺口都必须列明并报错，不靠补整日 NaN 或缩短尾部绕过。`source_date_coverage` 记录请求/可用/有效起止、排除的前置日期和全部缺失日期；`rebuilt_date_coverage` 再记录重建后的实际日期与差异。两份源覆盖预检通过后，备份并强制按当前规则重建两份 DataBlock，避免旧版已把 Inf 变成 NaN 的块被跳过。仍在保存前后检查，完成后以新 ID 请求 `gru_15mraw` 训练。预检失败不重建、不提交训练；预检后的失败保留备份和已经完成的结果。

`20261018_repair_15mraw_inf_train_03` 替代已停用的 `_train_02`。它复用备份、按需重建、保存前后校验及训练请求流程，在保存前逐日检查重建结果中的每一个 Inf（包括正负无穷），读取与生产加载器相同的 `trade_ts/15min` 或备用来源，记录股票、日期、0..15 的 bar、特征、原始文件、原始值、dtype、amount/volume 和 OHLCV。完整证据写入备份目录下的 `<frame>_inf_evidence.jsonl`；`report.json` 的 `inf_audit` 按特征、来源、可复现机制计数，日志/邮件包含各类样例和汇总。

来源分为原始文件已有 Inf、有限原始值转 float32 溢出、无法解释的预处理异常。对原始 VWAP Inf，用同一条存储记录的 amount/volume 核对非零除零、非有限操作数及数值溢出；“可复现”不冒充历史供应商计算过程的证明，无法追溯的上游原因明确标为 `upstream_origin_unresolved`，原始文件和数值仍保留供进一步调查。缺少原始行、冲突重复行、非标准 bar 标签或无法解释的有限值/NaN 变 Inf 会使任务失败，不覆盖该块、不提交训练。除下述 VWAP 回退外，已定位到原始文件的 Inf 和可验证的转换溢出转为 NaN，原有 NaN、有限值和原始行情文件不修改；所有旧校验仍然执行，全 NaN 特征也不能通过。正常无需重建的块已经过无 Inf 检查，不重复审计原始库。

分钟聚合的 `trade_min_reform` 使用 float64 汇总成交额/量；无成交（volume=0）或 VWAP 商为 Inf 时取同一聚合 bar 的 close。`15mraw` prepro 在 float32 转换后也对零成交量或 Inf VWAP 使用同一 bar 的 close，兼容历史原始 Inf 和转换溢出。close 非有限时保留 NaN，不引入新的 Inf。其他缺失值和字段不因这项回退改变。

Autofix 在 prepro 回退前通过观察回调记录每年分块的 VWAP Inf 证据，报告字段为 `vwap_fallback_audits`，文件为 `<frame>_vwap_<start>_<end>_inf_evidence.jsonl`；所以被 close 替换的 Inf 仍有原始来源和公式核对记录。重建完成后再审计其余 Inf 并转 NaN。此代码更新不重写历史原始行情。成功只代表 DataBlock 修复完成和训练请求已提交，不代表服务器训练已经完成。

`20261017_repair_15mraw_train_02` 检查 `15mraw` 的 fit/predict 两份 DataBlock：缺失、不可读、旧 `turn_fl` 特征、全 NaN 特征或无穷值会触发重算；特征和数值检查通过则跳过，不因数据日期较旧而重算。仅重建 `15mraw`，不改原始行情或其他 DataBlock。受影响的原文件/目录先复制到 `data/Interim/autofix_backups/15mraw_*/`，按年读取原始行情重建至本次固定的数据截止日，保存前检查完整交易日覆盖，保存前后校验特征、数值与保存后的坐标。`report.json` 记录每份数据的原因、备份和执行阶段；失败即停止，保留备份及已经完成的结果，不自动回滚。`dry_run: true` 只检查、不写入。修复代码与该清单须一同发布，已消耗的 ID 需要换新 ID 才能再次运行。

两份数据通过检查（包括本来正常、无需重算）后，脚本为 `gru_15mraw` 以当前 Autofix ID 写入训练请求（手动执行时使用脚本默认 ID）。Autofix 隔离环境没有 GPU，因此由已启用的 `idle_worklist` 服务在维护锁释放、GPU 空闲后优先启动训练，沿用工作列表的训练参数（当前 `resume: false`）。请求仅改变该模型的完成版本，不重置其他模型，同一请求 ID 不会重复触发已完成的训练；异常恢复仍遵循训练服务现有策略。`dry_run` 或数据修复/校验失败不会提交训练请求。数据修复报告的 success 只表示数据完成；训练状态和结果见既有 worklist/训练日志及邮件。请求文件保留在 `.local_resources/<机器名>/schedule_worklist/requests/`，不能在完成后删除，否则版本回退会引起额外训练。此任务使用新 ID 替代原先仅重算数据的 `_datablocks_01`，避免修改已冻结声明。

`20261015_rename_label_datablocks_01` 仅将 fit/predict 下的旧 `X_y_*` 改名为 `Y_*`，支持 `.mmap` 目录、`.pt` 和 `.feather` 文件，不读取或重新计算数据。执行前检查全部命名冲突，新旧名称并存时停止且不覆盖；已改名的文件再次检查会跳过。迁移记录位于 `data/Interim/autofix_reports/`，部分失败时可据此核对路径；若需要由 autofix 再次执行，仍需新 ID。读取暂时兼容旧名称，新写入统一使用新名称。

- **每个服务器、项目实例内，每个 ID 最多尝试一次。** 成功、失败、脚本不存在、参数错误、启动失败、超时或结果不明，均不自动重试。
- **重新执行必须追加不同 ID。** 没有 revision、reset、force 或清空历史后重跑的入口。
- 等待其他任务结束只是排队，不算尝试。首次登记后执行声明被冻结；修改同一 ID 的脚本、参数或超时会产生冲突，不能让旧 ID 重跑。
- 一条失败不阻塞其他 ID。存在先后依赖的多个步骤请放进同一个脚本，脚本自行在失败处停止。
- 正常进入脚本后，完成邮件由脚本负责，autofix 强制传入 `email: true`。Autofix 不再补发普通成功或业务失败邮件。
- 每个 ID 正式领取时，watchdog 另发一次启动流程通知，不受 `params.email` 影响；纯排队不发。

## 文件怎么写

初始内容：

```yaml
schema_version: 1
fixes: []
```

将来需要修复时，修改为下面的结构。**以下是格式示例，不是已准入、可直接执行的任务**：

```yaml
schema_version: 1
fixes:
  - id: 20261014_repair_prices_01
    description: 修复指定区间的价格数据
    script: scripts/2_data/repair_prices.py
    params:
      start: 20200101
      end: 20251231
      email: true
    timeout: 12h
    enabled: true
```

字段说明：

- `id`：必填，唯一且稳定。以英文字母或数字开头，之后可含字母、数字、下划线、点和短横线，最多 128 字符。建议使用日期、用途和编号。
- `script`：必填，项目根目录下 `scripts/` 内的 Python 相对路径。拒绝绝对路径、`..`、符号链接和未准入脚本。
- `params`：参数映射，可省略表示使用脚本的静态默认值。未知参数、框架保留参数、错误类型会导致启动失败。
- `params.email`：可省略或写 `true`；autofix 总是强制传入 `true`，旧条目的布尔值 `false` 也会被覆盖。字符串等非布尔值仍视为参数错误。
- `description`：可选说明，修改不会重新执行。
- `enabled`：默认 `true`。`false` 撤回尚未尝试的请求，不中断已经开始的任务。
- `timeout`：默认 `12h`，支持正数加 `s`、`m`、`h`、`d`，例如 `30m`、`12h`。包括进程初始化、计算及结束阶段；超时后先 TERM，30 秒后 KILL。

每条任务按文件顺序进入队列；不同 ID 可以指向相同脚本和相同参数，这被认为是两次明确的执行请求。整个文件变化、无关 Git 提交和文件排序变化不会重跑已尝试的 ID。

## 脚本如何准入

脚本需要已有项目的 `ScriptTool` 入口，并在开头 YAML 注释中显式添加 `autofix: true`。例如：

```python
# description: Repair prices
# autofix: true
# parameters:
#   start:
#     type: int
#     default: 20200101
#   end:
#     type: int
#     required: true

from src.proj.util.script import ScriptTool

@ScriptTool('repair_prices')
def main(start: int = 20200101, end: int = 20251231, **kwargs):
    # 校验输入 -> 备份原数据 -> 生成新数据 -> 校验结果 -> 发布
    # 失败应抛出异常，不能只打印“失败”后正常返回。
    ...

if __name__ == '__main__':
    main()
```

脚本必须是一个明确、有限的修复动作，只调用一次主入口。参数必须有静态的 `str`、`int`、`float`、`bool` 类型，或字面量选项列表，例如 `[true, false]`。首版不接受动态 `Options.*()` 类型、任意表达式、复合参数或隐式求值。头部参数应与 `main` 的显式参数一致。`email` 是框架公共选项，不需要在头部重复声明。

校验和预览不会导入脚本，也不会运行其顶层代码。`source`、`task_id`、`email_recipient`、`forfeit_if_done` 等框架控制参数不能从清单覆盖。执行来源标记为 `autofix`，不沿用日更的“今天已完成就跳过”逻辑。

准入意味着你已经检查了脚本的业务安全性：输入范围、原文件备份、结果校验、异常传播以及中途失败后的数据状态。**Autofix 不代替这些检查，也不禁止项目内所有删除操作。** 清理缓存、删除临时文件仍由脚本负责。

## 日常维护：只在本电脑编辑

1. 新修复：新增 ID，指定准入脚本和参数。
2. 发布前运行 `validate` 和 `plan`，确认没有格式、参数或路径错误。
3. 提交并推送到服务器自动拉取的分支；目前项目使用 `origin/master`。
4. 服务器下一轮发现新请求，等待维护时机后尝试一次。
5. 失败或需要再算一次：复制原条目，**改成新 ID**，调整参数，再发布。

不要仅修改失败条目的参数并期待重试。服务器可能已经见过旧声明，而你本机不知道；最可靠的习惯是发布后修改执行内容就换新 ID。

修改说明或调整顺序是安全的。删除条目只撤回尚未尝试的请求，不删除账本、不撤销数据变化。已尝试的 ID 即使删除后重新加入，也不会运行。未尝试的 ID 重新加入相同声明可以重新排队。

两台服务器有各自的账本：相同 ID 在每台已启用服务器上各最多尝试一次。不要把某台服务器的账本当作另一台机器的执行证明。

## 校验、预览与状态

在项目根目录使用项目 Python 环境：

```bash
python -m src.api.task_monitor.scheduling.autofix validate
python -m src.api.task_monitor.scheduling.autofix plan
python -m src.api.task_monitor.scheduling.autofix status
```

三者均为只读操作，不初始化账本，不执行修复。`plan` 展示解析后的实际参数（含默认值），`status` 在服务器上结合本机账本展示记录。本机没有服务器账本时会显示 `unknown (server ledger unavailable)`，这不等于服务器尚未执行。

常见状态：`queued` 等待、`withdrawn` 已撤回或声明冲突、`starting` 启动准备、`running` 运行、`success` 成功、`failed` 失败、`unknown` 结果待核查。`attempted: true` 表示此 ID 永远不能再次领取。

启动记录先持久化，再创建进程。若服务器恰好在这两步之间崩溃，可能出现“未实际执行但 ID 已消耗”；这是为了保证最多一次。反过来，已经修改数据但没来得及保存成功记录的任务，也不会被盲目重跑。根据日志和数据现场判断后，另加新 ID。

## 邮件与日志

- **开始启动流程**：领取 ID 的同一事务保存 `starting` 邮件事件，watchdog 在下一次通知轮询中发送，正常约一分钟内；内容包括 ID、服务器、领取时间、Git commit、脚本、脱敏参数及日志位置。这表示已经开始启动准备，不代表业务已开始或成功。即使后续参数校验失败或领取后崩溃，通知事件仍保留。失败只重试邮件，不重跑任务；上线前已经尝试的旧记录不补发启动通知。
- **脚本接管后正常结束或业务报错**：由脚本自己的邮件机制处理，autofix 强制启用其 email 参数，保留内容和附件；autofix 不代发第二封结果邮件。
- **脚本未接管**：autofix 发送启动失败邮件，注明 ID、服务器、提交、脚本、脱敏参数、阶段、异常、有限错误输出和日志路径，并注明必须新建 ID 才能重试。创建了 Python 进程不等于业务已开始。
- **超时、异常信号、进程丢失或重启**：由监督机制发异常中断告警；不作为普通业务失败再发第二封结果邮件。
- **整份 YAML 格式错误、重复 ID**：按文件内容去重告警，暂停新请求发现；不虚构任务执行结果。

通知失败只重试通知，绝不重跑脚本。脚本结果邮件沿用现有邮件队列；autofix 告警使用独立持久队列。SMTP 在“已经发出、发送回执尚未持久化”时崩溃可能产生重复邮件，不能保证邮件绝对一次。

`email` 现在是 autofix 的强制通知策略，优先于脚本头部默认值和旧清单的 `email: false`。比较冻结声明时，仅将缺省/布尔 false/true 统一为 true，兼容旧账本；其他执行参数仍保持冻结规则。仅修改这个布尔设置不会产生冲突，也不会清除已尝试状态。已执行或正在运行的任务不会因此重跑、补发结果邮件或改变当前进程。

执行账本、完整监督日志和通知事件位于 `~/.local/state/learndl/<项目路径摘要>/autofix/`。`status` 会显示实际位置和每次日志路径。脚本还保留原有任务记录、报告和邮件附件。

## 调度与互斥

Watchdog 只做短检查，长任务由独立 `learndl-autofix.service` 执行。补跑定时任务由 `learndl-deferred.service` 执行，保留普通任务的运行环境。两种 worker 互斥，补跑队列优先。普通受管任务持有共享维护许可，autofix 持有独占许可。修复等待期间不再启动新的空闲训练；已有任务自然结束。新到期的定时任务持久排队，不因维护锁忙而丢弃；先补跑到期任务，再处理 autofix 和空闲训练。

已存在的日更、周更、回补 shell 入口在启用本功能后转入相同 runner。普通任务自身原有锁仍有效。直接启动未接入 `ScriptTool`／训练生命周期的自定义程序、旧进程或外部编辑器不受保证；启用前应让旧任务结束。自动 Git 更新也参与维护互斥，不会在修复运行中换代码。

## 服务器首次安装与更新

只在 Linux 服务器配置执行，本机 macOS 可编辑、校验和预览。服务器需要 systemd、Bubblewrap，以及可用的非特权 user namespace；Bubblewrap 必须支持 `--disable-userns`，该选项由上游 [0.8.0](https://github.com/containers/bubblewrap/releases/tag/v0.8.0) 引入。应使用发行版仍维护且提供该能力的软件包；仅安装某些旧发行版默认提供的版本可能不够。安装后，使用现有调度安装器：

```bash
bash runs/install_schedules.sh plan --host mengkjin-server --diff
bash runs/install_schedules.sh apply --host mengkjin-server
bash runs/install_schedules.sh status --host mengkjin-server
```

以现有普通服务器账号运行安装器，不要整体使用 sudo。安装器仅为受管 systemd 单元使用已有 sudo 安装步骤。`maintenance.yaml` 已包含 autofix job，但**拉取 Git 不等于安装服务**。首次 apply 初始化账本并安装 worker；后续任务清单增量变化不需要再次 apply。隔离策略或 systemd 配置变化后需要重新 plan/apply。

已经安装旧 watchdog 的服务器也必须执行一次上述 `plan/apply`，无需先卸载。单独重启 watchdog 不会创建 worker、账本或隔离策略。安装器会和 worker、维护许可互斥；有任务占用时应等待结束后再 apply，而不是强行切换配置。启用状态保存在项目外的账本中，项目内配置被误删不能关闭互斥保护。

如果安装报错 `Bubblewrap must support --disable-userns` 或 `lacks --disable-userns`，先在执行安装器的同一账号、同一环境下检查：

```bash
cat /etc/os-release
command -v bwrap
bwrap --version
bwrap --help
```

这表示当前找到的程序缺少所需能力，不是 tasks.yaml 错误。优先通过当前发行版的官方更新仓库或受支持的 backports 获取新版；若已升级却仍报错，检查 PATH 是否仍优先选择旧副本。升级后确认帮助中出现 `--disable-userns`，再执行 `plan/apply`。不要删除这个检查或执行参数、清空账本，或关闭系统安全策略来绕过此错误。安装器按实际能力检测，支持发行版回移该选项的版本；不只比较版本号。

清单为空时 worker 不执行脚本。缺少隔离组件或隔离失败会拒绝执行，不退回普通 Python 执行。不可直接在终端运行内部 `worker` 代替已安装服务。

`rollback` 撤回安装配置，不撤销修复的数据，也不删除执行历史。不要在有运行任务或待补跑定时任务时卸载维护服务。

服务器查询及回滚命令：

```bash
systemctl status learndl-autofix.timer learndl-autofix.service learndl-deferred.timer learndl-deferred.service
journalctl -u learndl-autofix.service -u learndl-deferred.service --since today
python -m src.api.task_monitor.scheduling.autofix status
bash runs/install_schedules.sh rollback --host mengkjin-server
```

`status` 的 `history` 包含已从清单移除的 ID，`scheduled` 显示定时补跑记录。补跑启动后监督进程崩溃的记录可能停留在 `starting`，应结合普通任务账本及日志人工核查，不自动重复启动。

如果看起来“空闲但未运行”，先查询 `status` 和 worker 的 journal。`activation_enabled` 表示项目外账本中的启用开关；`last_worker_check` 给出最近一次 worker 检查时间及等待原因，例如补跑队列优先、维护许可被占用、旧脚本记录待核查，或没有可尝试的新 ID。时间长期不更新时优先检查 timer/service；旧版尚无此字段时显示 null，不代表从未运行。`dispatching` 也可能正在进行隔离预检，不能据此判断业务已经启动。不要通过删除账本、清锁或修改旧 ID 绕过等待。

## 文件隔离范围

宿主文件系统默认只读，项目根目录可写；`.venv`、`.git`、`.secret` 是只读保护例外，避免改动共享依赖、版本元数据和凭据。账本、安装后的隔离策略在项目外，不可由脚本改写。临时目录和进程空间独立，子进程继承限制。

执行前检查嵌套挂载、特殊文件和指向可写范围外的硬链接。项目外的数据目录不会自动获得写权限，路径布局不兼容就失败。Python 环境可能使用 uv 硬链接，因此 `.venv` 必须保持只读。扫描在 worker 内进行，不阻塞 watchdog。

Linux 挂载检查读取 mountinfo，包含同文件系统的 bind mount；无法读取完整目录时拒绝执行。监督进程访问项目内通信文件时不跟随符号链接，避免通过通知文件间接写到项目外。启动前已持久化 systemd 进程范围，创建进程到登记 PID 之间崩溃时也必须确认没有残留子进程，才恢复其他任务。

默认允许 TCP/IP 联网；服务限制宿主 Unix 控制 socket，沙箱没有宿主设备的任意访问权。需要 Unix socket 或 GPU 设备的脚本不能假定兼容，应先做隔离验收。

保护的是本次任务及其子进程的直接文件访问，不保证业务结果正确，不阻止通过网络修改外部系统。整个项目中的业务代码可写，因此被改动代码以后在普通环境运行的间接影响也不在此边界内。不要把它当作运行恶意脚本的系统。

## 账本备份与故障恢复

账本只在显式首次安装时创建。已初始化后缺失或损坏，会暂停自动执行；重新 apply 不能用空账本替代历史。备份整个状态目录，SQLite 数据库应在 worker 停止、没有写入时复制，或使用 SQLite backup API。

恢复时先停止 worker/timer，恢复同一服务器和项目的完整备份，再核对最后日志与数据现场。若备份不包含最近已执行的任务，不能直接恢复自动调度，否则失去最多一次保证。不要删除状态目录“清理缓存”。项目目录迁移会改变项目标识，需要保留并核对旧账本后再迁移，不能当作全新实例自动重跑。

## 验证

本机契约测试：

```bash
python -m unittest src.api.task_monitor.tests.test_autofix -v
```

Linux 隔离验收（只操作测试创建的临时文件，不运行清单中的修复）：

```bash
LEARNDL_AUTOFIX_SANDBOX_TESTS=1 python -m unittest src.api.task_monitor.tests.test_autofix.IsolationTests -v
```

还应在服务器用一次性的测试脚本和独立测试项目，验收 systemd cgroup 清理、超时、维护期间定时任务排队、脚本业务失败邮件和启动失败邮件。macOS 上通过的单元测试不能替代 Linux/systemd 验收。
