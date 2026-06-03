# UV 安装和使用指南

本指南将帮助您在Windows和Ubuntu 22.04系统上安装uv，并在learndl项目中创建Python 3.12虚拟环境，安装所有依赖包。

## 目录
- [系统要求](#系统要求)
- [安装UV](#安装uv)
- [创建虚拟环境](#创建虚拟环境)
- [安装依赖](#安装依赖)
- [本地包安装](#本地包安装)
- [常见问题](#常见问题)

## 系统要求

- **Python版本**: 3.12
- **操作系统**: Windows 10/11 或 Ubuntu 22.04 LTS
- **项目**: learndl (已部署)

## 安装UV

### Windows系统

#### 方法1: 使用PowerShell安装 (推荐)
```powershell
# 以管理员身份运行PowerShell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm https://astral.sh/uv/install.ps1 | iex
```

#### 方法2: 使用winget安装
```cmd
winget install --id=astral-sh.uv
```

#### 方法3: 手动下载安装
1. 访问 [UV Releases](https://github.com/astral-sh/uv/releases)
2. 下载最新的Windows x64版本
3. 解压到 `C:\Program Files\uv\` 目录
4. 将 `C:\Program Files\uv\` 添加到系统PATH环境变量

### Ubuntu 22.04系统

#### 方法1: 使用官方安装脚本 (推荐)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 方法2: 使用pip安装
```bash
pip install uv
```

#### 方法3: 使用apt安装 (如果可用)
```bash
sudo apt update
sudo apt install uv
```

### 验证安装
```bash
# 检查uv版本
uv --version

# 检查uv帮助
uv --help
```

## 创建虚拟环境

### 1. 进入项目目录
```bash
cd /path/to/learndl  # 替换为您的实际项目路径
```

### 2. 创建Python 3.12虚拟环境
```bash
# 创建虚拟环境
uv venv --python 3.12

# 激活虚拟环境
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Ubuntu/Linux
source .venv/bin/activate
```

### 3. 验证虚拟环境
```bash
# 检查Python版本
python --version  # 应该显示 Python 3.12.x

# 检查pip版本
pip --version

# 检查虚拟环境路径
which python  # Ubuntu/Linux
where python  # Windows
```

## 安装依赖

### 方法1: 使用uv pip install (推荐用于现有项目)
```bash
# 确保在项目根目录且虚拟环境已激活
uv pip install -r requirements.txt
```

### 方法2: 使用uv add生成pyproject.toml (推荐用于新项目或现代化)
```bash
# 首先创建pyproject.toml文件
uv init

# 然后逐个添加依赖包
uv add numpy==1.26.0
uv add pandas==2.3.0
uv add scipy==1.15.3
uv add matplotlib==3.10.3
uv add scikit-learn==1.7.0
uv add torch==2.7.1
uv add streamlit==1.46.0

# 或者批量添加多个包
uv add numpy==1.26.0 pandas==2.3.0 scipy==1.15.3 matplotlib==3.10.3

# 添加开发依赖
uv add --dev pytest black flake8
```

### 方法3: 从requirements.txt迁移到pyproject.toml
```bash
# 1. 创建pyproject.toml
uv init

# 2. 读取requirements.txt并添加依赖
while IFS= read -r line; do
    # 跳过空行和注释
    if [[ ! -z "$line" && ! "$line" =~ ^# ]]; then
        # 移除版本号中的==，uv add会自动处理
        package=$(echo "$line" | sed 's/==.*//')
        version=$(echo "$line" | sed 's/.*==//')
        uv add "$package==$version"
    fi
done < requirements.txt

# 3. 验证安装
uv sync

# 4. 查看生成的文件
ls -la pyproject.toml uv.lock
```

### 4. 使用uv sync管理依赖
```bash
# 安装pyproject.toml中定义的所有依赖
uv sync

# 安装开发依赖
uv sync --dev

# 更新依赖到最新版本
uv sync --upgrade

# 清理未使用的依赖
uv sync --clean
```

### 2. 分批安装依赖 (如果遇到问题)
```bash
# 先安装基础包
uv pip install numpy pandas scipy matplotlib

# 再安装机器学习相关包
uv pip install scikit-learn xgboost lightgbm catboost

# 安装深度学习相关包
uv pip install torch torchvision torchaudio

# 安装其他依赖
uv pip install -r requirements.txt
```

### 3. 验证安装
```bash
# 检查已安装的包
uv pip list

# 测试关键包导入
python -c "import numpy, pandas, torch; print('所有包导入成功!')"
```

## 依赖管理方法对比

### uv pip install vs uv add 的区别

| 特性 | uv pip install | uv add |
|------|----------------|---------|
| **配置文件** | 使用requirements.txt | 生成pyproject.toml |
| **依赖解析** | 传统pip解析 | 现代化依赖解析器 |
| **锁定文件** | 不生成 | 自动生成uv.lock |
| **版本管理** | 固定版本 | 支持版本范围 |
| **开发依赖** | 混合管理 | 分离管理 |
| **兼容性** | 完全兼容pip | 现代化标准 |

### 选择建议

#### 使用 uv pip install 的情况：
- 项目已有requirements.txt且运行正常
- 团队习惯pip工作流程
- 需要快速部署，不想重构依赖管理
- 项目依赖相对简单

#### 使用 uv add 的情况：
- 新项目或重构项目
- 需要现代化的依赖管理
- 需要精确的依赖锁定
- 需要分离生产和开发依赖
- 团队愿意采用新的工作流程

### pyproject.toml 的优势

```toml
[project]
name = "learndl"
version = "0.1.0"
description = "深度学习量化交易系统"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26.0",
    "pandas>=2.3.0",
    "torch>=2.7.1",
    "streamlit>=1.46.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## 本地包安装

### 1. 识别本地安装包
根据您的需求，以下包可能需要本地安装：

#### 金融数据包
```bash
# 安装rqdatac (需要本地程序)
# 确保rqdatac的本地程序已安装并配置

# 安装tushare (可能需要配置token)
uv pip install tushare
```

#### 优化求解器
```bash
# Mosek (商业软件，需要许可证)
# 确保Mosek已安装并配置环境变量

# CVXOPT相关
uv pip install cvxopt cvxpy
```

#### 其他特殊包
```bash
# 安装可能需要编译的包
uv pip install --no-binary :all: some-package

# 安装开发版本
uv pip install --editable .
```

### 2. 配置本地包
```bash
# 创建配置文件
mkdir -p ~/.config/uv
touch ~/.config/uv/config.toml

# 编辑配置文件 (根据实际路径调整)
echo '[tool.uv]
python = "3.12"
venv = ".venv"' > ~/.config/uv/config.toml
```

## 项目配置

### 1. 更新项目脚本
修改 `runs/` 目录下的脚本，使用虚拟环境：

```bash
# 在脚本中添加虚拟环境激活
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    BASE_PATH="/home/mengkjin/workspace/learndl"
    PYTHON_CMD="/home/mengkjin/workspace/learndl/.venv/bin/python"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    BASE_PATH="/Users/mengkjin/workspace/learndl"
    PYTHON_CMD="/Users/mengkjin/workspace/learndl/.venv/bin/python"
fi
```

### 2. 创建激活脚本
```bash
# 创建activate.sh
cat > activate.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "虚拟环境已激活: $(which python)"
echo "Python版本: $(python --version)"
EOF

chmod +x activate.sh
```

## 常见问题

### 1. 权限问题
```bash
# Ubuntu/Linux
sudo chown -R $USER:$USER .venv/
chmod +R 755 .venv/

# Windows
# 以管理员身份运行PowerShell
```

### 2. 网络问题
```bash
# 使用国内镜像源
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或者配置pip.conf
mkdir -p ~/.pip
echo "[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn" > ~/.pip/pip.conf
```

### 3. 包冲突
```bash
# 清理环境重新安装
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate  # Ubuntu/Linux
# 或 .venv\Scripts\Activate.ps1  # Windows
uv pip install -r requirements.txt
```

### 4. 内存不足
```bash
# 分批安装
uv pip install --no-cache-dir -r requirements.txt

# 或者使用系统包管理器预安装一些包
# Ubuntu
sudo apt install python3-numpy python3-pandas python3-scipy
```

## 维护和更新

### 1. 更新依赖
```bash
# 更新所有包到最新版本
uv pip install --upgrade -r requirements.txt

# 更新特定包
uv pip install --upgrade package-name
```

### 2. 导出依赖
```bash
# 导出当前环境的所有依赖
uv pip freeze > requirements_current.txt

# 导出特定包的依赖
uv pip show package-name
```

### 3. 清理缓存
```bash
# 清理uv缓存
uv cache clean

# 清理pip缓存
uv pip cache purge
```

## 总结

通过本指南，您应该能够：
1. 在Windows和Ubuntu 22.04上成功安装uv
2. 在learndl项目中创建Python 3.12虚拟环境
3. 安装requirements.txt中的所有依赖包
4. 配置本地程序相关的包
5. 解决常见的安装和使用问题

如果在安装过程中遇到任何问题，请检查：
- Python版本是否为3.12
- 虚拟环境是否正确激活
- 网络连接是否正常
- 系统权限是否足够
- 本地程序是否正确安装和配置
