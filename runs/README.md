# Runs 文件夹说明

本文件夹包含了用于自动化运行和启动learndl项目的Shell脚本。

## 文件结构

```
runs/
├── launch.sh              # 应用启动脚本（使用智能配置）
├── daily_update.sh        # 每日更新脚本
├── weekly_update.sh       # 每周更新脚本
├── computer_config.sh     # 计算机配置文件（新增）
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
- **参数**：`--source=bash --email=1`
- **功能**：自动化执行每日数据更新，支持邮件通知
- **使用场景**：定时任务、每日维护

### 2. `weekly_update.sh` - 每周更新脚本
**作用**：执行每周数据更新任务
- **目标脚本**：`src/scripts/1_autorun/1_weekly_update.py`
- **参数**：`--source=bash --email=1`
- **功能**：自动化执行每周数据更新，支持邮件通知
- **使用场景**：定时任务、每周维护

### 3. `launch.sh` - 应用启动脚本
**作用**：启动learndl的Streamlit应用
- **目标**：启动Streamlit Web应用
- **参数**：`--server.runOnSave=True`
- **功能**：启动Web界面，支持代码热重载
- **使用场景**：开发调试、应用部署

### 4. `computer_config.sh` - 计算机配置文件（新增）
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
# 每日凌晨2点执行
0 2 * * * /path/to/learndl/runs/daily_update.sh

# 每周日凌晨3点执行
0 3 * * 0 /path/to/learndl/runs/weekly_update.sh
```

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
