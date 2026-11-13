# Syncthing 研究结果投递与本机归档

服务器脚本按需发布完成的分析结果，Mac 每 300 秒接收。Syncthing 负责跨网络传输；本模块负责快照、校验、归档和清理。服务器原件保留，不自动传输模型权重、检查点或原始训练数据。autofix 上下文跳过，不扩大沙盒权限。

已经完成一次 100 MiB 中国服务器 → 美国 Mac 传输及 SHA-256 校验。换网、休眠恢复和跨时段稳定性尚待验证。300 秒是本机归档轮询周期，不是跨境到达时限。

## 目录与协议

两端共享 **`$HOME/workspace/Syncthing`**，Folder ID 为 `learndl-artifacts`，类型必须是 **Send & Receive**，文件版本保留关闭。

```text
~/workspace/Syncthing/artifact_exchange/
  model_training/<完整模型名称>/<UTC时间>__<UUID>.transfer.zip
  model_compare/<实验名称>/<UTC时间>__<UUID>.transfer.zip
  factor_sequence/<实验名称>/<UTC时间>__<UUID>.transfer.zip
  rl/<实验名称>/<UTC时间>__<UUID>.transfer.zip
  <自定义研究类型>/<实验名称>/<UTC时间>__<UUID>.transfer.zip
```

新包使用 schema v2，按产生结果的研究类型及实验合包，不再使用顶层 reports/tensorboard/optuna。研究类型是最多 48 位的小写字母、数字、下划线标识，字母开头；保留目录名不能作为类型。实验原名和电脑原名完整写入清单，路径中的特殊字符或超长名称转换时加哈希，避免重名。

本机目标：

```text
results/model_training/server__nn@modelAAA/<UTC时间>__<UUID>/
  results/                 # 模型自身 ModelPath.rslt() 的完整文件树
  tensorboard/<名称>.zip
  optuna/<名称>.zip
  source.json
```

其他研究也遵循 `results/<研究类型>/<来源电脑>__<实验名称>/<批次>/`。没有产出的角色目录不创建。每个运行、每个实验一个包，包含各角色的成员大小和 SHA-256；一个任务有多个模型时分别发布。同一批次声明的源路径、目标路径、来源身份及运行 ID 构成不可变发布身份；修改内容后必须使用新 run_id。重试复用已准备的快照。直接写入共享区的普通文件不会被消费，手工材料也请使用 publish CLI。

`source_host` 是发布端配置中的稳定电脑名，init 默认取当时主机名，可指定 `--source-host server`。不要跟随 Wi-Fi 的 DHCP 主机名修改它。旧配置未包含该字段时首次发布把主机名保存在私有 state/source_host.json，后续继续使用；建议显式补入配置。

实验面板新增“同步研究实验”，依据已提交批次的 source.json 识别 TensorBoard/Optuna ZIP，展示电脑、研究类型、实验、批次时间和状态。普通报告 ZIP 不进入实验列表。本机模型、数据库、旧 `results/model_experiments/` ZIP 仍可打开；缺少来源信息的旧 ZIP 显示“来源未知”。新接收器仍接收 schema v1 并保持其旧归档路径，不重新搬动历史文件。

默认每层 ZIP 解压限制为 4 GiB、50,000 成员，配置的 `max_bytes` / `max_files` 同时用于接收与面板导入，外层清单计入限制。需要更大产物时两端同步调整配置并检查磁盘空间。包内已登记的所有文件完整归档，不按扩展名二次过滤。

服务状态、不可变待发包、错误记录、接收回执和轮转日志位于项目 `.local_resources/artifact_exchange/`，不共享。原子发布使用共享根目录旁的临时目录；归档暂存位于目标文件系统。接收依次校验、落盘、记录回执、删除共享包；删除经 Syncthing 回传。服务器“published”只代表本地共享发布成功，不等于远端收到。进程中断可能重发同一个包，接收端幂等处理。

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

更新项目代码、使用已有项目虚拟环境。在**各自项目根目录**运行以下命令。核心 CLI 只依赖 Python 标准库，不要求加载模型或机器密钥。

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

嵌套任务通过 contextvars 隔离登记。模型训练使用独立模型生命周期收集器：完整 rslt()、已关闭 TensorBoard、关联 Optuna 在模型锁内完成快照并发布，把交接记录交给外层任务处理邮件；同一模型多次运行不会到任务末尾才复制已经改变的文件。正常恢复走相同出口，保存失败时不发布不完整快照。

只有模型自身 results/ 及调用者明确声明的分析目录全量复制，不扫描整个模型目录或项目 results。factor_sequence、RL 等仍按既有分析选择器排除权重、检查点、原始数据与恢复材料。它们登记的报告及实验载荷在同一批次归档；分析选择器仍采用明确的后缀/排除规则。新脚本应主动登记完整的纯分析目录，不能直接声明含权重的旧 bundle 或运行根目录。autofix 环境始终跳过。

只有成功发布的对应分析附件才从当次邮件移除。HTML 执行报告和日志继续发送；RL 的分析交接记录会替代旧大 bundle 附件，原始 bundle 仍留服务器。发布失败保留邮件原行为，不改变任务业务状态。已准备的历史邮件重试记录不重写。

## 7. 历史报告迁移

只在 Mac receiver 配置下运行。关闭正在写这些报告的本地任务：

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
