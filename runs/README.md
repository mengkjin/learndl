# Runs 文件夹说明

本文件夹包含了用于自动化运行和启动learndl项目的Shell脚本。

## Shell脚本写作方式

### 1. 脚本结构模式
所有脚本都遵循相同的结构模式，包含以下几个部分：

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
- **错误处理**：包含完整的错误检查和提示
- **环境验证**：运行前验证必要环境
- **路径管理**：使用绝对路径避免路径问题

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

### 定时任务
```bash
# 添加到crontab实现自动化
# 每日凌晨2点执行
0 2 * * * /path/to/learndl/runs/daily_update.sh

# 每周日凌晨3点执行
0 3 * * 0 /path/to/learndl/runs/weekly_update.sh
```

## 注意事项

1. **权限要求**：确保脚本有执行权限
2. **环境依赖**：需要正确安装Python和依赖包
3. **路径配置**：根据实际环境修改BASE_PATH
4. **错误处理**：脚本包含完整的错误检查机制
5. **跨平台**：支持Linux和macOS系统
