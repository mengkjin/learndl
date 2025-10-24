可以在Ubuntu上实现这个监控程序。以下是完整的解决方案：

## 方法1：使用Shell脚本和cron

### 1. 创建监控脚本

创建一个脚本文件，比如 `/usr/local/bin/system_monitor.sh`：

```bash
#!/bin/bash

# 监控脚本
MONITOR_FILE="/path/to/your/monitor.file"  # 替换为实际文件路径
SCRIPT_DIR="/path/to/your/scripts"         # 替换为实际脚本目录

# 步骤1: 检查监控文件是否存在
if [ ! -f "$MONITOR_FILE" ]; then
    exit 0
fi

# 步骤2: 检查是否有指定目录下的脚本正在运行
scripts_running=false

# 获取所有正在运行的进程，排除grep本身和当前脚本
for pid in $(ps -eo pid --no-headers); do
    # 检查进程的命令行是否包含脚本目录路径
    if grep -q "$SCRIPT_DIR" /proc/$pid/cmdline 2>/dev/null; then
        scripts_running=true
        break
    fi
done

# 如果没有脚本在运行，执行步骤3
if [ "$scripts_running" = false ]; then
    echo "No scripts from $SCRIPT_DIR are running. Removing monitor file and rebooting..."
    rm -f "$MONITOR_FILE"
    sudo reboot
fi
```

### 2. 设置脚本权限

```bash
sudo chmod +x /usr/local/bin/system_monitor.sh
```

### 3. 配置cron任务

编辑cron表：
```bash
sudo crontab -e
```

添加以下行：
```bash
0 * * * * /usr/local/bin/system_monitor.sh
```

## 方法2：使用systemd定时器（推荐）

### 1. 创建监控脚本

同方法1，但需要稍作修改以更好地与systemd集成：

```bash
#!/bin/bash

# 监控脚本 - systemd版本
MONITOR_FILE="/path/to/your/monitor.file"
SCRIPT_DIR="/path/to/your/scripts"

log() {
    echo "$(date): $1"
}

# 步骤1: 检查监控文件是否存在
if [ ! -f "$MONITOR_FILE" ]; then
    exit 0
fi

log "Monitor file found: $MONITOR_FILE"

# 步骤2: 检查是否有指定目录下的脚本正在运行
scripts_running=false

# 更精确的进程检查方法
while IFS= read -r pid; do
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        if grep -q "$SCRIPT_DIR" /proc/$pid/cmdline 2>/dev/null; then
            log "Script from $SCRIPT_DIR is running (PID: $pid)"
            scripts_running=true
        fi
    fi
done < <(ps -eo pid --no-headers)

if [ "$scripts_running" = false ]; then
    log "No scripts from $SCRIPT_DIR are running. Removing monitor file and rebooting..."
    rm -f "$MONITOR_FILE"
    sync
    systemctl reboot
fi
```

### 2. 创建systemd服务文件

创建 `/etc/systemd/system/system-monitor.service`：

```ini
[Unit]
Description=System Monitor Service
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/system_monitor.sh
User=root
```

### 3. 创建systemd定时器文件

创建 `/etc/systemd/system/system-monitor.timer`：

```ini
[Unit]
Description=Run system monitor hourly
Requires=system-monitor.service

[Timer]
OnCalendar=*-*-* *:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### 4. 启用并启动定时器

```bash
sudo systemctl daemon-reload
sudo systemctl enable system-monitor.timer
sudo systemctl start system-monitor.timer
```

## 验证和测试

### 检查定时器状态
```bash
# 对于systemd方法
sudo systemctl status system-monitor.timer

# 查看所有活跃的定时器
systemctl list-timers
```

### 测试脚本
```bash
# 手动运行测试
sudo /usr/local/bin/system_monitor.sh
```

## 重要注意事项

1. **权限设置**：确保脚本有适当的执行权限，并且cron或systemd以root权限运行

2. **文件路径**：替换脚本中的文件路径和目录路径为实际值

3. **安全考虑**：自动重启系统可能影响服务，请确保在测试环境中验证

4. **日志记录**：建议添加更详细的日志记录以便调试

5. **防误删**：可以在删除文件前添加确认步骤或备份

推荐使用systemd方法，因为它提供更好的日志集成、错误处理和系统集成。