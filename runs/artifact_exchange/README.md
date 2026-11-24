# Syncthing 研究结果投递与本机归档

服务器脚本按需发布完成的分析结果，Mac 每 300 秒接收。Syncthing 负责跨网络传输；本模块负责快照、校验、归档和清理。服务器原件保留，不自动传输模型权重、检查点或原始训练数据。autofix 上下文跳过，不扩大沙盒权限。

已经完成一次 100 MiB 中国服务器 → 美国 Mac 传输及 SHA-256 校验。换网、休眠恢复和跨时段稳定性尚待验证。300 秒是本机归档轮询周期，不是跨境到达时限。

## 目录与协议

两端共享 **`$HOME/workspace/Syncthing`**，Folder ID 为 `learndl-artifacts`，类型必须是 **Send & Receive**，文件版本保留关闭。

```text
~/workspace/Syncthing/artifact_exchange/
  models/<完整模型名称>/<UTC时间>__<UUID>.transfer.zip
  model_compare/<实验名称>/<UTC时间>__<UUID>.transfer.zip
  factor_sequence/<实验名称>/<UTC时间>__<UUID>.transfer.zip
  rl/<实验名称>/<UTC时间>__<UUID>.transfer.zip
  <自定义研究类型>/<实验名称>/<UTC时间>__<UUID>.transfer.zip
```

新包使用 schema v2，按产生结果的研究类型及实验合包，不再使用顶层 reports/tensorboard/optuna。研究类型是最多 48 位的小写字母、数字、下划线标识，字母开头；保留目录名不能作为类型。实验原名和电脑原名完整写入清单，路径中的特殊字符或超长名称转换时加哈希，避免重名。

本机目标：

```text
results/external/models/server/nn@modelAAA/<build>/
  results/                 # 模型分析报告
    tensorboard/<名称>.zip
    optuna/<名称>.zip
  html/<run_id>.html       # 显式训练或测试的运行记录
  logs/                    # property.jsonl、train.md 和 test.md
  build.json               # 文件校验、更新顺序和运行索引
```

其他研究也遵循 `results/external/<研究类型>/<来源电脑>/<实验名称>/<批次>/`。没有产出的角色目录不创建。每个运行、每个实验一个包，包含各角色的成员大小和 SHA-256；一个任务有多个模型时分别发布。同一批次声明的源路径、目标路径、来源身份及运行 ID 构成不可变发布身份；修改内容后必须使用新 run_id。重试复用已准备的快照。直接写入共享区的普通文件不会被消费，手工材料也请使用 publish CLI。

`source_host` 是发布端配置中的稳定电脑名，init 默认取当时主机名，可指定 `--source-host server`。不要跟随 Wi-Fi 的 DHCP 主机名修改它。旧配置未包含该字段时首次发布把主机名保存在私有 state/source_host.json，后续继续使用；建议显式补入配置。

实验面板新增“同步研究实验”，依据批次的 source.json 或模型的 build.json 识别 TensorBoard/Optuna ZIP，展示电脑、研究类型、实验、批次时间和状态。普通报告 ZIP 不进入实验列表。本机模型、数据库、旧 `results/custom/legacy/model_experiments/` ZIP 仍可打开；缺少来源信息的旧 ZIP 显示“来源未知”。接收器兼容旧版 `reports/<类别>/` 路由和模型包的 `model/`、`model_training/` 路由，旧在途包也归入新归档目录。

默认每层 ZIP 解压限制为 4 GiB、50,000 成员，配置的 `max_bytes` / `max_files` 同时用于接收与面板导入，外层清单计入限制。需要更大产物时两端同步调整配置并检查磁盘空间。包内已登记的所有文件完整归档，不按扩展名二次过滤。

服务状态、不可变待发包、错误记录、接收回执和轮转日志位于项目 `.local_resources/artifact_exchange/`，不共享。原子发布使用共享根目录旁的临时目录；归档暂存位于目标文件系统。接收依次校验、落盘、记录回执、删除共享包，然后清理协议路径中的空实验目录和空类别目录；每轮也清理历史残留空目录。保留 artifact_exchange、共享根目录、符号链接及任何非空目录（包括含隐藏文件或同步临时文件的目录）。删除经 Syncthing 回传。服务器“published”只代表本地共享发布成功，不等于远端收到。进程中断可能重发同一个包，接收端幂等处理。

## 1. 安装 Syncthing

### 中国 Ubuntu/Debian 服务器

以运行研究的普通用户登录，仅以下安装和服务管理步骤使用 sudo：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl python3
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fSL https://syncthing.net/release-key.gpg -o /etc/apt/keyrings/syncthing-archive-keyring.gpg
sudo chmod 0644 /etc/apt/keyrings/syncthing-archive-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/syncthing-archive-keyring.gpg] https://apt.syncthing.net/ syncthing stable-v2' | sudo tee /etc/apt/sources.list.d/syncthing.list
sudo apt-get update
sudo apt-get install -y syncthing
sudo systemctl enable --now "syncthing@$(id -un).service"
systemctl status "syncthing@$(id -un).service" --no-pager
syncthing device-id
```

只运行 Syncthing 服务；没有服务器投递定时任务。服务进程以普通用户运行。

### 美国 Mac

```bash
brew install syncthing
brew services start syncthing
syncthing device-id
open http://127.0.0.1:8384
```

如系统尚未安装 Homebrew，先按 https://brew.sh 的官方安装说明安装。为 GUI 设置用户名和密码，监听地址保留 localhost。安装来源：[官方 APT](https://apt.syncthing.net/)、[Homebrew](https://formulae.brew.sh/formula/syncthing)、[自动启动](https://docs.syncthing.net/users/autostart.html)。

## 2. 设备配对与正式共享

两端执行 `syncthing --version`，当前验证版本为 2.1.7。

新机器：交换 `syncthing device-id` 输出后，在每端执行一次：

```bash
ST_ARTIFACT_PEER_ID='对端的完整Device-ID'
syncthing cli config devices add --device-id "$ST_ARTIFACT_PEER_ID" --name Research-Peer
```

目前两台电脑已配对，跳过添加设备。测试共享 `learndl-connectivity-probe` 中的文件已确认双向删除，Mac 的测试共享配置和目录也已移除；服务器空目录及共享配置需在服务器端清理。以下 ID 来自本次实际配对；更换设备时重新读取。

服务器先设置：

```bash
ST_ARTIFACT_PEER_ID='P4QQS2A-3EHPYOJ-OCVRYIO-ND5RZXN-VLLAVXS-IKUIJML-TXQVYW7-CUCC6QU'
```

Mac 设置：

```bash
ST_ARTIFACT_PEER_ID='OEQGWUG-N76R3WV-FIZMTPK-3F3N7E4-W6YLVTJ-JUIWQZU-H4LY3C6-PJTQ6AM'
```

然后两端分别执行：

```bash
mkdir -p "$HOME/workspace/Syncthing/artifact_exchange"
syncthing cli config folders add --id learndl-artifacts --label Learndl-Artifacts --path "$HOME/workspace/Syncthing"
syncthing cli config folders learndl-artifacts type set sendreceive
syncthing cli config folders learndl-artifacts devices add --device-id "$ST_ARTIFACT_PEER_ID"
syncthing cli config options global-ann-enabled set true
syncthing cli config options relays-enabled set true
syncthing cli config options natenabled set true
syncthing cli config folders learndl-artifacts dump-json
syncthing cli show connections
```

添加命令仅执行一次；若已从 GUI 接受同一 Folder ID，核对路径/共享设备后跳过重复添加。**`natenabled` 没有连字符**。仅共享以上目录，不共享项目、配置目录或密钥。文件夹 `.stfolder` 由 Syncthing 创建，应用不伪造它。

双方无公网入站地址也可通过公共中继连接。保留默认发现及中继；不要为本功能开放 GUI 的 8384 端口。若已有 SSH，可用 `ssh -L 9999:127.0.0.1:8384 用户@服务器入口` 访问远端管理界面。直连通常使用 TCP/UDP 22000，中继以实际节点端口为准。参见[网络说明](https://docs.syncthing.net/users/firewall.html)及[中继说明](https://docs.syncthing.net/users/relaying.html)。

## 3. 配置应用角色

更新项目代码、使用已有项目虚拟环境。在**各自项目根目录**运行以下命令。Linux 和 macOS 上的核心 CLI 只依赖 Python 标准库，Windows 文件锁使用项目已有的 portalocker。

服务器：

```bash
.venv/bin/python -m src.automation.artifact_exchange init --role publisher --enable --source-host server
```

Mac：

```bash
.venv/bin/python -m src.automation.artifact_exchange init --role receiver --enable
```

`init` 不覆盖现有配置。已有文件时直接编辑 `.local_resources/artifact_exchange/config.json`。未创建配置或 `enabled=false` 时功能关闭，邮件路径维持原行为。路径必须是展开后的绝对路径；也支持 `~` 和环境变量。自定义配置用 CLI 的全局 `--config /absolute/config.json`，或脚本环境变量 `LEARNDL_ARTIFACT_CONFIG`。

```bash
.venv/bin/python -m src.automation.artifact_exchange doctor
```

确认角色、路径、enabled 和 folder_marker。不要在服务器设置 receiver；不要把 results 或私有 state 放入共享根目录。

### 从旧版投递代码升级

先将本次代码更新部署到两端；服务器尚未由本机远程更新，不能把本机验收当作服务器部署成功。已有正式共享和设备配对保持原样，不重复执行 folders add，也不重新迁移历史结果。

服务器已有 publisher 配置时，在服务器项目根目录执行一次，明确固定电脑名：

```bash
.venv/bin/python - <<'PYCONFIG'
import json
from src.automation.artifact_exchange.core import DEFAULT_CONFIG, atomic_json
values = json.loads(DEFAULT_CONFIG.read_text())
assert values['role'] == 'publisher', '请先核对服务器角色'
values.setdefault('source_host', 'server')
atomic_json(DEFAULT_CONFIG, values)
print('source_host:', values['source_host'], 'enabled:', values['enabled'])
PYCONFIG
.venv/bin/python -m src.automation.artifact_exchange doctor
```

若配置不存在，使用上面的 publisher init 命令。更新后的下一次研究执行会使用新批次协议；旧待发快照仍可 retry，接收器兼容 v1。Mac 的 launchd 每次启动新的 Python 进程，无需改变五分钟周期；长期运行的实验面板须从控制台重启，再刷新浏览器页面加载新来源。

## 4. 首次手动验收

在服务器创建独立测试报告（目录位于项目外，原件保留）：

```bash
mkdir -p "$HOME/workspace/artifact-smoke-source"
printf 'artifact smoke test\n' > "$HOME/workspace/artifact-smoke-source/report.txt"
.venv/bin/python -m src.automation.artifact_exchange publish "$HOME/workspace/artifact-smoke-source" --task smoke --run-id smoke-001 --category custom --experiment smoke
```

等待 Syncthing 同步，在 Mac 执行：

```bash
.venv/bin/python -m src.automation.artifact_exchange receive --once
.venv/bin/python -m src.automation.artifact_exchange status
```

核对输出 target 下 `results/report.txt`、`source.json`；确认两端共享包最终删除，服务器源目录保留。重复发布同一身份不再产生副本。修改报告后使用 `--run-id smoke-002`。

TensorBoard 手动发布示例：

```bash
.venv/bin/python -m src.automation.artifact_exchange publish /absolute/model/snapshot/tensorboard --task training --run-id training-001 --kind tensorboard --category model_training --experiment nn@modelAAA --name modelAAA
```

Optuna 改为数据库路径及 `--kind optuna`，会使用 SQLite backup。每个研究目录应在写入器关闭、异步保存完成后发布。

## 5. Mac 每五分钟接收

```bash
.venv/bin/python -m src.automation.artifact_exchange service plan
.venv/bin/python -m src.automation.artifact_exchange service install
.venv/bin/python -m src.automation.artifact_exchange service status
```

LaunchAgent 为 `com.learndl.artifact-exchange`，固定当前解释器、项目和配置的绝对路径；登录执行一次，之后每 300 秒运行，有进程锁防重叠。电脑休眠期间不会主动唤醒，恢复后补收。更换虚拟环境或移动项目后重新 install。

接收日志：`.local_resources/artifact_exchange/receiver.log`，每份 2 MB，保留三份旧日志。上次结果在 `last_receive.json`，长期成功回执在 `received/`。

## 6. 研究脚本接口与邮件行为

```python
from pathlib import Path
from src.automation.artifact_exchange import ArtifactSpec
from src.automation.artifact_exchange.research import publish_safe

# 写盘与 writer.close() 完成之后；全部来源是绝对路径。
specs = [
    ArtifactSpec(Path('/absolute/model/results'), category='model_training',
                 experiment='nn@modelAAA'),
    ArtifactSpec(Path('/absolute/model/snapshot/tensorboard'), kind='tensorboard',
                 category='model_training', experiment='nn@modelAAA', name='training'),
    ArtifactSpec(Path('/absolute/study.sqlite3'), kind='optuna',
                 category='model_training', experiment='nn@modelAAA', name='search'),
]
records = publish_safe('train_models', 'unique-run-001', specs, status='success')
```

`publish_safe` 保证传输错误不掩盖研究异常。底层 `publish_artifacts` 提供同样的参数及可选 `config=`，适合 CLI 或需要自行处理异常的调用方。

`ArtifactSpec` 的字段：`source`、`kind`、`category`、`experiment`、`name`、`members`、`destination`、`run_id`。`kind` 是 reports/tensorboard/optuna；reports 表示普通结果文件，不产生 reports 中间目录。默认目标是 results/ 或同名实验角色目录；`destination` 可指定批次内的安全相对子目录。`members` 是相对 source 的明确文件清单；未指定时完整复制声明目录。每项可用 run_id 分开同一任务中的不同运行，否则继承发布调用的 run_id。重叠目标拒绝发布，不覆盖合并。

自定义脚本使用现有任务上下文：

```python
# 在现有 with AutoRunTask(...) as task: 内登记；任务退出等待异步保存后统一发布。
task.register_artifact(ArtifactSpec(
    Path('/absolute/completed-analysis'), category='custom_research',
    experiment='experiment_A', destination='results',
))
```

嵌套任务通过 contextvars 隔离登记。模型训练使用独立模型生命周期收集器：完整 rslt()、已关闭 TensorBoard、关联 Optuna 在模型锁内先生成不可变待发快照；在 AutoRunTask 内运行时，此时只准备，不写入共享目录。同一模型多次运行不会到任务末尾才复制已经改变的文件。正常恢复走相同出口，保存失败时不准备不完整快照。

只有模型自身 results/ 及调用者明确声明的分析目录全量复制，不扫描整个模型目录或项目 results。factor_sequence、RL 等仍按既有分析选择器排除权重、检查点、原始数据与恢复材料。它们登记的报告及实验载荷在同一批次归档；分析选择器仍采用明确的后缀/排除规则。新脚本应主动登记完整的纯分析目录，不能直接声明含权重的旧 bundle 或运行根目录。autofix 环境始终跳过。

AutoRunTask 的统一收尾顺序为：等待异步保存 → 发布已登记产物和已准备快照 → 通过任务 logger 输出一次投递数量汇总（prepared/published/skipped/failed），失败原因去重后最多展示三种，不逐包列出 ID、实验和目标路径 → 关闭 HTML/Markdown 日志捕获 → 发送结果邮件和任务邮件。这样投递结果包含在邮件日志中。因子验证、逐训练结果、RL 等内部邮件在任务内先排队，返回 queued 状态，到投递完成后才选择附件并发送；各自的持久化邮件状态/重试记录仍保留。独立调用、没有 AutoRunTask 的脚本维持立即投递后发邮件。

没有待投递产物、发布未启用、receiver 角色和 autofix 的正常跳过保持静默，也不在邮件正文重复提示。内部逐包投递记录仍保留；实际发布、失败、异步保存未完成，以及已准备包被阻止发送均计入任务汇总。邮件正文也仅展示汇总。

每日 resume_testing 沿用模型 full_name 和服务器 results 目录；每次调用仍产生独立 run_id，接收端在同一 `external/models/<host>/<experiment>/<build>/` 下按文件更新，沿用未变化的结果，并在 `build.json` 中记录投递的运行。模型权重保存在发布端。

快照准备中断后可以通过 CLI retry 发布；定时接收不负责服务器投递。投递失败使用 warning 记录，不加入研究任务的 error conclusions。

只有成功发布的对应分析附件才从当次邮件移除。HTML 执行报告和日志继续发送；RL 的分析交接记录会替代旧大 bundle 附件，原始 bundle 仍留服务器。发布失败保留邮件原行为，不改变任务业务状态。已准备的历史邮件重试记录不重写。

## 7. 0.7.4 之前的历史报告迁移

以下为旧版本操作记录。当前版本请使用第 12 节的版本补丁。

```bash
.venv/bin/python -m src.automation.artifact_exchange migrate-results plan
.venv/bin/python -m src.automation.artifact_exchange migrate-results apply
```

已有历史文件本次不重迁。新 schema v2 归档目录不允许被历史迁移命令搬走。

映射：model、gru_day_rtn → model_training；model_compare → model_compare；storage_reports → storage；coverage_diag → data_quality。归入 `results/reports/<类别>/legacy/<原目录>`，旧路径留下相对兼容链接。若已存在 `results/model_compare` → `reports/model_compare/legacy/model_compare` 的旧链接，新接收器只兼容这一精确的迁移目标，新批次通过原入口可见；不会移除链接或搬动旧文件。其他目标符号链接仍拒绝。RL、factor_sequence、nn_training_review、data_repairs 等运行目录保留。

apply 检查 lsof 打开的写句柄、最近五分钟写入、前后清单和校验和；状态不确定或目标冲突时跳过。仍须避免同时启动写报告的新任务。迁移记录在 state/migrations 下。回滚仅在内容未改变时允许：

```bash
.venv/bin/python -m src.automation.artifact_exchange migrate-results rollback
```

## 8. 故障恢复与卸载

- 服务器断网：发布仍可完成，Syncthing 后续传输；没有服务器网络等待任务。
- 共享 marker 缺失：发布保留不可变待发快照；修复 Syncthing 配置后执行 `retry`。
- 打包前失败：failures 记录原因；修复源文件后重新调用原发布命令。
- 损坏包、路径冲突、磁盘不足：接收记录错误，包保留，不覆盖用户修改。修复原因后再次 `receive --once`。
- 进程中断：重试相同身份，接收回执和来源清单防重复；不要手工删除 state 作为修复方式。
- `status` 可查看已发布、待发记录和最近接收结果；没有服务器收到远端成功回执的承诺。

```bash
.venv/bin/python -m src.automation.artifact_exchange retry
.venv/bin/python -m src.automation.artifact_exchange service uninstall
```

卸载接收任务保留数据、配置及回执。停止自动发布将服务器配置 `enabled` 改为 false。删除正式 Syncthing folder 配置前先确认没有未收取包。后续如需重新进行联网稳定性测试，请另建独立测试共享。

## 9. 邮件档案和模型预览

更新两端代码后先用 `doctor` 核对角色、共享路径和固定的 `source_host`。本机继续使用
现有接收任务。重启项目控制台后可启动 **邮件档案**（8506）和 **模型预览**（8507）。
端口可分别由 `LEARNDL_MAIL_ARCHIVE_PORT` 和 `LEARNDL_MODEL_PREVIEW_PORT` 覆盖。
两个服务监听本机地址，远程访问使用现有的 SSH 端口转发方式。

- 启用 artifact 发布的服务器默认归档项目邮件，包括日常更新、模型训练和测试、Watchdog 通知
  及直接通过项目 SMTP 接口发送的邮件。可在脚本
  YAML 头部设置 `mail_sync: true` 或 `false`，也可在 `ScriptTool`、`AutoRunTask`
  或邮件入口显式传入。SMTP 仍使用原有配置。邮件档案中的“SMTP 未知”表示接收时
  没有 SMTP 回执，发送与重试进度仍由发件端的原有队列记录。
- 档案含逻辑通知的 EML、正文、SMTP 附件与产物交接前的原附件。邮件及附件长期保留。
  归档只改变列表显示。删除进入本机应用回收站，恢复或彻底删除在该视图操作。
- 0.7.7 起，模型本地的 `logs/train.md` 和 `logs/test.md` 按实际运行阶段追加。
  接收机将变更文件合入同一 build，累计日志保存在 `results/external/models/<来源电脑>/<模型>/<build>/logs/`。
  从头训练会分配新的 build，恢复训练和测试沿用原 build。
- 显式训练、手动恢复训练和独立测试将本次模型 HTML 保存到 `logs/html/<run_id>.html`。
  下次模型运行或产物收集时，历史 `run_logs/html` 自动合入该目录；同名异内容保留带摘要后缀的历史副本。
  已有父任务完整 HTML 时不再重复附加模型 HTML 到邮件，但仍通过 artifact 归档。自动模型更新和
  `resume_testing` 仍可由父脚本保存一份任务 HTML，模型调用只写 Markdown。
  分析 callback 的报告作为独立结果打开。模型预览和实验面板共享已启动的实例和额度。
- 模型预览和 CLI 的模型归档操作共用 `src/res/model/workflow/model_management`。
  原有管理历史、移动和撤销日志继续从控制台状态目录读取，升级时保留该目录。
- 启用后可用一次正常的小型任务验收。确认本机能打开邮件的原始附件、按邮件关联
  找到模型运行章节，并从两个面板打开同一 TensorBoard 或 Optuna 实例。

## 10. 0.7.2 目录迁移与历史邮件补投

空环境会以当前代码版本建账并跳过这些历史补丁。迁入旧环境的文件时，先恢复原版本台账。
如果没有台账，先用 `python -m src.automation.version_patch init --baseline <数据版本>`
登记已确认的数据版本，再查看迁移计划。

升级前，在将要运行补丁的 checkout 中用项目解释器查看只读预检：

```bash
PYTHONPATH=. .venv/bin/python scripts/.version_patch/0.7.2/support/result_layout.py
PYTHONPATH=. .venv/bin/python scripts/.version_patch/0.7.2/support/mail_backfill.py
.venv/bin/python -m src.automation.version_patch plan
```

先更新接收机，再更新服务器。按 [版本补丁说明](../../scripts/.version_patch/README.md)
在空闲窗口运行补丁。迁移前关停正在阅读或写入实验文件的 TensorBoard、Optuna 和其他程序。
本机接收程序会通过接收锁与迁移互斥。升级后重启控制台和实验面板。

- `0.7.2/001` 将模型的 TensorBoard 移到 `results/tensorboard/`，将可唯一归属到模型的
  Optuna SQLite 备份到 `results/optuna/`。旧 SQLite 和 WAL 保留供恢复。
  同步批次内的 ZIP 同样移到 `results/{tensorboard,optuna}/`，原清单和接收回执不变。
  若报告显示冲突或文件正在使用，处理原因后重试该补丁。
- `0.7.2/002` 在已启用的发布端固定首次执行时刻之前七天的窗口，补投有 SMTP 接受凭据的
  日常数据和模型任务邮件。原始 EML、留档附件和可关联的分卷一并打包。
  接收机跳过发布；服务器不会重发 SMTP 邮件。若报告显示待投递，恢复共享目录后使用
  `python -m src.automation.artifact_exchange retry`，本机再核对接收档案。
- 两个补丁在各自的版本补丁尝试目录写入 JSON 报告。历史发送前就移除的附件无法恢复；
  无法确认所属模型的旧数据库保留在独立实验入口。

## 11. 0.7.3 模型结果浏览与残差数据目录

模型预览以来源和模型为单位列出文件夹。选择模型后可预览结果文件与日志，也可打开
TensorBoard 和 Optuna。XLSX 预览可切换工作表，每张显示前 40 行和 20 列。

`0.7.3/001` 把模型目录（含旧模型归档）和已接收的 `model_training` 批次中的
`results/factor_residual/` 移到同级 `snapshot/factor_residual/`。训练以后将相关中间
数据直接写入 snapshot。旧版在途包在接收时映射到新路径，原收据和校验清单仍然有效。
如果目标已存在或相关文件仍被占用，补丁停止并在报告中指出该目录；处理后用版本
补丁的 retry 命令继续。升级后重启模型预览服务，以启用 XLSX 预览接口。


## 12. 0.7.4 两端 results 目录

项目结果目录固定为 `test`、`trade`、`factor`、`gp`、`custom`、`external`。
本机生成的通用报告、运维诊断和研究工作目录写入 `results/custom/<用途>/`。
项目代码用 `PATH.rslt_custom`，显式扫描其他项目根目录的工具用
`src.proj.env.path.custom_results(root)`。磁盘报告的 HTML 和同名 SQLite 索引一起保存在
`custom/storage_reports/`。服务器的 RL 工作目录是 `custom/rl_experimental/`。

接收端使用 `external/<类别>/<来源电脑>/<实验>/<批次>/`。模型的投递类别仍填写
`model_training`，新共享包放入 `artifact_exchange/models/<完整模型名>/`。
RL 批次归入 `rl_experiment`，因子序列修复批次归入 `factor_sequence`。
模型归档目录为 `external/models/`，已有归档由 0.7.5 补丁迁移。0.7.7 补丁将旧批次和 `run_logs/` 合并到对应 build。未能确认来源的历史材料保存在 `custom/legacy/`。

先在接收机更新，再更新服务器。在两端的项目根目录预检并走现有版本补丁流程：

```bash
PYTHONPATH=. .venv/bin/python scripts/.version_patch/0.7.4/support/results.py
.venv/bin/python -m src.automation.version_patch plan
.venv/bin/python -m src.automation.version_patch apply
```

迁移清单逐项记录源路径、目标和内容校验值，并更新接收回执和面板恢复路径。
出现冲突时先核对两处文件，保留需要的副本，再用
`python -m src.automation.version_patch retry 0.7.4/001` 继续。升级后重启模型预览、
实验、邮件和存储面板。旧 RL 请求、配置、身份文件使用原字节进行校验。

### 存储洞察日报

服务器的 Daily Update 完成且数据已更新到目标日期后，会使用 publisher 配置生成静态存储日报。
每个服务器当地自然日最多成功发布一次，固定文件为
`$HOME/workspace/Syncthing/persistent/storage_insights/<source_host>/latest.html`，其中目录名沿用发布器的安全主机名编码。
日报复用现有扫描统计和展示深度，只生成 HTML，完整写好后原子替换。
失败保留上一份并写入 Daily Update 日志，同日再次执行可重试。
部分目录不可读时报告显示部分统计及错误数量。

本机“存储洞察”提供本机和服务器来源选择，记住上次选择。
服务器报告以扫描时间和已有目录明细为准，新版同步到达时点击“查看新版”切换。
打开子服务或切换来源不会自动扫描本机；本机扫描由页面按钮触发。发布端在页面点击重新扫描后，会把刚生成的报告通过同一发布流程强制覆盖到 `latest.html`，跳过当天已发布检查。
该功能沿用 Daily Update 调度，更新代码并重启本机存储洞察服务即可，无需新增 watchdog 安装。
