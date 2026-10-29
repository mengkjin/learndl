# Runs 文件夹说明

本文件夹包含了用于自动化运行和启动learndl项目的Shell脚本。

## 项目磁盘占用报告

在项目 CLI 中选择 **Non-Research Operations → Project Storage Report**，输入扫描根目录和 HTML 输出路径；直接回车使用生产项目目录与默认报告路径。

也可在项目根目录运行（默认发送邮件附件）：

```bash
uv run --frozen python -c 'from src.api.calls.computer import ProjectStorageReport; ProjectStorageReport.go(root=".")'
```

`root` 支持绝对路径；`output` 可指定 HTML 文件路径，相对路径基于扫描根目录。默认输出到 `results/storage_reports/storage_<timestamp>.html`，已有文件不会覆盖。传入 `email=False` 可只保存本地报告；`recipient="you@example.com"` 可覆盖默认收件人，否则沿用项目邮箱配置（发送给发件邮箱）。发送失败仍保留报告，并沿用现有邮件队列的重试机制。

报告可离线浏览，支持目录折叠、路径搜索和同级排序。数据库展示到数据集、模型展示到模型名称，中间数据和导出分别展示到子用途和因子/模型/组合；更深层文件仍计入合计。按实际分配磁盘块统计，包含隐藏目录，不跟随软链接，硬链接只计一次；出现读取错误时会标记不完整。仅支持提供 `st_blocks` 的系统（如 Linux/macOS）。扫描期间文件可能变化，报告不是文件系统快照，也不会删除任何文件。

### 按需查看下一层目录

新报告会在 HTML 旁保存同名 `.sqlite3` 索引。索引来自首次扫描，保留文件和文件夹的占用；邮件只附加 HTML，不附加索引。默认层级内同时展示文件和文件夹，文件行不提供展开按钮。

报告默认隐藏名称以 `.` 或 `~` 开头的文件和文件夹（包括其后代）。头部可取消勾选隐藏选项，显示这些条目；该设置同时适用于搜索和动态加载，只改变展示，不影响统计总量或占比。

交互运行 **Non-Research Operations → Project Storage Report** 后，在生成报告和邮件处理结束时，会询问“是否在本地浏览器打开报告？”。选择是后启动本机服务并自动打开浏览器；选择否或取消则直接结束。只有标准输入、输出均为交互终端时才询问，定时任务或重定向运行不会等待输入。

浏览器打开终端打印的 `http://127.0.0.1:<port>/<session>/` 网址。在默认最底层且仍有文件或子文件夹的节点名称右侧，点击“加载下一层”即可继续查看；每次只加载一层，加载过的内容会复用。显示的仍是首次扫描时的大小，不会重新扫描磁盘。搜索、排序适用于已加载文件和文件夹，全部展开只展开已加载的文件夹。

服务仅监听本机，使用随机端口与会话地址；按 Ctrl+C 停止并结束任务。如果无法自动打开浏览器，可在同一计算机上手动打开打印的网址。服务停止后，页面已加载内容仍可查看。直接打开 HTML 或邮件附件保持静态模式，不会连接任何本机服务。旧报告未保存文件明细，需重新生成才能看到文件。

HTML 与索引需要一起保留，旧报告、索引缺失或损坏、报告被修改或与索引不匹配时，需要重新生成。索引包含生成机器指纹；Linux 使用机器 ID，其余系统使用主机名和网卡标识摘要，因此更改这些机器标识后也需重新生成报告。删除报告时可同时删除其同名 `.sqlite3` 文件。

## 文件结构

```
runs/
├── launch.sh              # 应用启动脚本（使用智能配置）
├── launch.bat             # 应用启动脚本（Windows version）
├── daily_update.sh        # 每日更新脚本
├── weekly_update.sh       # 每周更新脚本
├── computer_config.sh     # 计算机配置文件（新增）
├── install_watchdog.sh    # 统一安装器的兼容入口
├── install_schedules.sh   # 调度 plan/apply/status/rollback
├── scheduling/            # 任务、时间、维护和服务器配置
└── README.md              # 本说明文档
```

## Shell脚本写作方式

### 1. 脚本结构模式
所有脚本都遵循相同的结构模式，包含以下几个部分：

#### 1.0 计算机配置系统（新增）
```bash
# 引入计算机配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/computer_config.sh" ]; then
    source "$SCRIPT_DIR/computer_config.sh"
fi

# 获取配置
if command -v get_computer_config >/dev/null 2>&1; then
    CONFIG=$(get_computer_config)
    BASE_PATH=$(echo "$CONFIG" | cut -d'|' -f1)
    PYTHON_CMD=$(echo "$CONFIG" | cut -d'|' -f2)
else
    # 回退配置
    BASE_PATH=$(dirname "$SCRIPT_DIR")
    PYTHON_CMD="python"
fi
```
- 作用：自动识别计算机名称并配置相应的路径和Python命令
- 功能：支持跨平台、跨网段的智能配置
- 优势：无需手动修改脚本，自动适应不同环境

#### 1.1 Shebang声明
```bash
#!/bin/bash
```
- 作用：指定脚本使用的解释器为bash
- 位置：脚本第一行

#### 1.2 操作系统检测
```bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux环境配置
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS环境配置
fi
```
- 作用：自动检测操作系统类型
- 功能：根据不同的OS设置相应的路径和Python命令
- 变量：`$OSTYPE` 系统内置变量，用于识别操作系统

#### 1.3 环境变量设置
```bash
BASE_PATH="/path/to/learndl"  # 项目根目录路径
PYTHON_CMD="python3"          # Python解释器命令
```
- 作用：定义关键路径和命令
- 跨平台：Linux和macOS使用不同的路径格式

#### 1.4 环境检查
```bash
# 检查Python解释器是否可用
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD interpreter not found"
    exit 1
fi

# 检查工作目录是否存在
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Directory $BASE_PATH does not exist"
    exit 1
fi
```
- 作用：验证运行环境
- 功能：检查Python是否安装、工作目录是否存在
- 错误处理：发现问题时输出错误信息并退出

#### 1.5 工作目录切换
```bash
cd "$BASE_PATH"
```
- 作用：切换到项目根目录
- 目的：确保脚本在正确的目录下执行

#### 1.6 主命令执行
```bash
$PYTHON_CMD [具体Python脚本路径] [参数]
```
- 作用：执行具体的Python脚本
- 参数：根据脚本需要传递相应的命令行参数

### 2. 脚本特点
- **跨平台兼容**：自动识别Linux和macOS
- **智能配置**：基于计算机名称自动配置路径和命令
- **短名称支持**：自动提取hostname中第一个点之前的部分，支持跨网段
- **错误处理**：包含完整的错误检查和提示
- **环境验证**：运行前验证必要环境
- **路径管理**：使用绝对路径避免路径问题
- **回退机制**：配置文件缺失时自动使用默认配置

## 脚本功能说明

### 1. `daily_update.sh` - 每日更新脚本
**作用**：执行每日数据更新任务
- **目标脚本**：`src/scripts/1_autorun/0_daily_update.py`
- **参数**：`--source=bash --email=1`；额外参数经 `"$@"` 透传（如 `--forfeit_if_done False` 强制重跑）
- **功能**：自动化执行每日数据更新，支持邮件通知
- **使用场景**：定时任务、每日维护

### 2. `weekly_update.sh` - 每周更新脚本
**作用**：执行每周数据更新任务
- **目标脚本**：`src/scripts/1_autorun/1_weekly_update.py`
- **参数**：`--source=bash --email=1`
- **功能**：自动化执行每周数据更新，支持邮件通知
- **使用场景**：定时任务、每周维护

### 3. `rcquant_sec_backfill.sh` - RCQuant 股票分钟线补全（可选）
**作用**：手动限幅向前补全 `trade_ts/min`（默认最多 5 个交易日）。**主路径已并入日更**：`RcquantMinBarDownloader` 在成功下载最新一天 sec 1min（且完成 etf/fut/cb）后自动触发。
- **目标脚本**：`scripts/1_autorun/5_rcquant_sec_backfill.py`
- **参数**：`--source=bash --email=1`，可选 `--force True --max_days 5`
- **无限回补**：`--max_days none`（或 `null`、显式空字符串）；仍受配额和午夜截止限制。CLI 参数提示直接回车使用默认值 5。
- **使用场景**：日更未触发时的补跑；21:01 定时任务显式使用无限天数。

### 4. `launch.sh` `launch.bat` - 应用启动脚本
**作用**：启动learndl的Streamlit应用
- **目标**：启动Streamlit Web应用
- **参数**：`--server.runOnSave=True`
- **功能**：启动Web界面，支持代码热重载
- **使用场景**：开发调试、应用部署

### 5. `computer_config.sh` - 计算机配置文件（新增）
**作用**：集中管理不同计算机的配置信息
- **功能**：自动识别计算机名称并配置相应的路径和Python命令
- **特点**：
  - 支持短名称识别（自动去掉域名部分）
  - 跨平台配置（Linux和macOS）
  - 函数式设计，易于扩展和维护
- **配置格式**：`BASE_PATH|PYTHON_CMD`
- **使用场景**：多计算机环境、跨网段部署

## 使用方法

### 直接执行
```bash
# 给脚本添加执行权限
chmod +x runs/*.sh

# 执行脚本
./runs/daily_update.sh
./runs/weekly_update.sh
./runs/rcquant_sec_backfill.sh
./runs/launch.sh
```

### 计算机配置管理
```bash
# 查看当前计算机信息
source runs/computer_config.sh
show_computer_info

# 列出所有已知计算机配置
list_known_computers

# 手动测试配置获取
get_computer_config

# 获取短名称
get_short_name
```

### 添加新的计算机配置
```bash
# 编辑 computer_config.sh 文件
# 在相应的case语句中添加新的计算机名

# Linux配置示例
case "$SHORT_HOSTNAME" in
    "your-linux-server")
        echo "/path/to/your/learndl|your-python-command"
        ;;
esac

# macOS配置示例
case "$SHORT_HOSTNAME" in
    "your-macbook")
        echo "/Users/username/workspace/learndl|/Users/username/workspace/learndl/.venv/bin/python"
        ;;
esac
```

### 定时任务
```bash
# 添加到crontab实现自动化
# 每日凌晨2点执行（当天已完成后 crontab 会 forfeit 跳过）
0 2 * * * /path/to/learndl/runs/daily_update.sh

# 强制重跑：忽略当天已完成记录
0 2 * * * /path/to/learndl/runs/daily_update.sh --forfeit_if_done False

# 每周日凌晨3点执行
0 3 * * 0 /path/to/learndl/runs/weekly_update.sh

# （可选）手动限幅补全 RCQuant sec 分钟线；日更已自动限幅补全，一般不需要 crontab
# 30 21 * * * /home/mengkjin/workspace/learndl/runs/rcquant_sec_backfill.sh
```

### Learndl Monitor / Watchdog 后台维护

监控页只读任务数据库；任务状态收敛、异常日志缓存、缓存清理和受管任务超时保护均由 watchdog 调度。请按 [统一调度安装说明](scheduling/README.md) 使用 `install_schedules.sh plan/apply`，配置文件支持保留有效旧 cron 并补充缺失计划。旧 cron 直接启动的任务不自动获得超时保护；新 runner 的日更默认 12 小时、周更 72 小时。不要为同一维护项另加独立 cron。

## 计算机配置系统详解

### 短名称识别机制
系统会自动提取hostname中第一个点之前的部分，支持跨网段部署：

```bash
# 示例：不同网段下的hostname
"ubuntu-desktop.local" → 短名称: "ubuntu-desktop"
"learndl-server.192.168.1.100" → 短名称: "learndl-server"
"MacBook-Pro-Mengkjin.home" → 短名称: "MacBook-Pro-Mengkjin"
"workstation-01.corp.company.com" → 短名称: "workstation-01"
```

### 配置查找流程
1. 获取完整hostname：`$(hostname)`
2. 提取短名称：`$(echo "$hostname" | cut -d'.' -f1)`
3. 根据操作系统类型和短名称查找配置
4. 解析配置字符串：`BASE_PATH|PYTHON_CMD`
5. 设置环境变量并执行

### 扩展配置
```bash
# 添加新的计算机配置
case "$SHORT_HOSTNAME" in
    "new-server"|"new-server-alias")
        echo "/opt/learndl|/opt/learndl/.venv/bin/python"
        ;;
esac
```

## 注意事项

1. **权限要求**：确保脚本有执行权限
2. **环境依赖**：需要正确安装Python和依赖包
3. **配置管理**：通过computer_config.sh管理不同计算机的配置
4. **短名称规则**：配置使用短名称（第一个点之前的部分）
5. **回退机制**：配置文件缺失时自动使用默认配置
6. **跨平台**：支持Linux和macOS系统
7. **跨网段**：支持不同网络环境下的hostname变化

## Autofix 数据修复

通过 Git 发布一次性脚本请求，服务器独立记账并在维护窗口内隔离执行。当前任务清单为空。每个 ID 最多一次尝试，重试必须新增 ID；配置、邮件规则和首次安装见 [Autofix 使用说明](autofix/README.md)。
