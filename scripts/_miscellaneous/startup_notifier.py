#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第 2 步：配置脚本
在运行脚本之前，您必须配置邮件发送相关的参数。
脚本默认会从环境变量读取配置，如果环境变量不存在，则使用代码中 SMTP_CONFIG 字典里的默认值。为了安全，强烈推荐使用环境变量来存储密码等敏感信息。
您有两种方式进行配置：
(推荐) 方式一：通过 systemd 服务文件设置环境变量
这是一种安全且灵活的方式，您将在下一步创建服务文件时看到如何设置。
(不推荐) 方式二：直接修改 Python 脚本
如果您只是快速测试，可以直接编辑 /usr/local/bin/startup_notifier.py 文件中的 SMTP_CONFIG 字典：

# src_runs/4_miscellaneous/startup_notifier.py

# ...
SMTP_CONFIG = {
    "server": "smtp.gmail.com",  # 您的 SMTP 服务器 (例如 Gmail)
    "port": 587,                 # SMTP 端口
    "sender_email": "your-email@gmail.com", # 您的发件邮箱
    "receiver_email": "destination-email@example.com", # 收件人邮箱
    "password": "your_google_app_password" # 您的邮箱密码或应用专用密码
}
# ...

> Gmail 用户请注意: 如果您使用 Gmail，需要生成一个 "应用专用密码" (App Password) 而不是使用您的常规登录密码，否则 Google 会阻止登录。
第 3 步：部署为 systemd 服务
为了让脚本在每次开机时自动运行，我们将其设置为一个 systemd 系统服务。
1. 将脚本移动到标准位置并授予执行权限
打开终端，执行以下命令：

sudo mv src_runs/4_miscellaneous/startup_notifier.py /usr/local/bin/
sudo chmod +x /usr/local/bin/startup_notifier.py

2. 创建 systemd 服务文件
执行以下命令来创建并编辑服务文件：

sudo nano /etc/systemd/system/startup-email.service
将以下内容完整地复制并粘贴到 nano 编辑器中：

[Unit]
Description=Send startup email notification
Documentation=https://github.com/mengkjin/learndl
# 确保在网络连接建立之后再运行此服务
After=network-online.target
Wants=network-online.target

[Service]
# 要执行的脚本路径
ExecStart=/usr/bin/python3 /usr/local/bin/startup_notifier.py

# (推荐) 在这里设置环境变量，以避免将密码硬编码到代码中
# 请取消下面的注释并替换为你自己的值
# Environment="SMTP_SERVER=smtp.gmail.com"
# Environment="SMTP_PORT=587"
# Environment="SMTP_SENDER=your-email@gmail.com"
# Environment="SMTP_RECEIVER=destination-email@example.com"
# Environment="SMTP_PASSWORD=your_app_password"

# 运行服务的用户和组。使用 root 可以确保有权限读取 journalctl 和写入 /var/log
User=root
Group=root

Type=oneshot
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target

重要: 如果您选择使用环境变量，请务必取消 Environment=行的注释（删除行首的#）并填入您的真实信息。完成后，按 Ctrl+X，然后按 Y，最后按 Enter 保存并退出。
3. 重新加载 systemd 并启用服务
执行以下命令使服务生效：


sudo systemctl daemon-reload
sudo systemctl enable startup-email.service

enable 命令会创建链接，确保服务在下次启动时自动运行。
第 4 步：测试
在重启电脑前，您可以手动测试脚本和服务是否工作正常。
1. 直接运行脚本测试
# 这会使用您在脚本中硬编码的配置（或未设置的环境变量）
sudo python3 /usr/local/bin/startup_notifier.py
检查您的收件箱是否收到了邮件，并查看终端输出的日志信息。
2. 测试 systemd 服务
# 这会使用您在 .service 文件中配置的环境变量来运行
sudo systemctl start startup-email.service

这个命令会立即启动一次服务。
3. 查看日志
如果邮件没有发送成功，可以通过以下命令查看服务的运行日志，定位问题：

# 查看 systemd 服务的日志
sudo journalctl -u startup-email.service -f

# 查看脚本自己记录的日志文件
cat /var/log/startup_notifier.log

"""

import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os

# get SMTP config from environment variables
SMTP_CONFIG = {
    "server": os.environ.get("SMTP_SERVER","smtp.163.com"),  
    "port": int(os.environ.get("SMTP_PORT" , 25)),                
    "sender_email": os.environ.get("SMTP_SENDER","mengkjin@163.com"), 
    "receiver_email": os.environ.get("SMTP_RECEIVER","mengkjin@163.com"), 
    "password": os.environ.get("SMTP_PASSWORD","TSkYh33f3pesHP2S") 
}

# some additional cmds to run after email sent
ADDITIONAL_CMDS = [
    '/home/mengkjin/workspace/learndl/runs/daily_update.sh'
]

# log file path
LOG_FILE = "/var/log/startup_notifier.log"

# log settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() 
    ]
)

def run_command(command):
    """safely execute shell command and return output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"command '{command}' failed: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        logging.error(f"command not found: {command.split()[0]}. please ensure the related tools (such as journalctl) are installed.")
        return None

def get_system_info():
    """get system startup and last shutdown info"""
    info = {
        "boot_time": "unknown",
        "last_shutdown_time": "unknown",
        "last_shutdown_status": "unknown",
        "abnormal_shutdown_reason": "none"
    }

    # 1. get current boot time
    boot_time_str = run_command("uptime -s")
    if boot_time_str:
        info["boot_time"] = boot_time_str

    # 2. analyze last boot log (journalctl -b -1)
    previous_boot_log = run_command("journalctl -b -1 --no-pager")
    
    if not previous_boot_log:
        info["last_shutdown_status"] = "insufficient info (maybe first boot or log has been rotated)"
        return info

    # 3. get last shutdown time (last log time of last session)
    # using short-iso format can get precise time with year and timezone
    last_log_time_str = run_command("journalctl -b -1 -n 1 --no-pager -o short-iso")
    if last_log_time_str:
        # format like: 2023-10-27T15:30:00+0800, we clean the format
        info["last_shutdown_time"] = last_log_time_str.split('+')[0].replace('T', ' ')

    # 4. check if last shutdown is normal
    if "Reached target Shutdown" in previous_boot_log or \
       "Reached target Power-Off" in previous_boot_log or \
       "Reached target Reboot" in previous_boot_log:
        info["last_shutdown_status"] = "normal shutdown"
    else:
        info["last_shutdown_status"] = "abnormal shutdown"
        # 5. if abnormal shutdown, try to analyze the reason
        # -p 3 means 'err' level and higher (crit, alert, emerg)
        error_logs = run_command("journalctl -b -1 -p err --no-pager -n 25") # get last 25 error logs
        if error_logs:
            info["abnormal_shutdown_reason"] = f"system log found error, last related error info:\n---\n{error_logs}\n---"
        else:
            info["abnormal_shutdown_reason"] = "no clear error(error) info found in last boot log. possible reasons include:\n" \
                                           "  - forced power off (e.g., directly unplug the power)\n" \
                                           "  - system frozen and forced reboot\n" \
                                           "  - kernel panic (Kernel Panic) but not recorded in log\n" \
                                           "  - some hardware failure"

    return info

def send_email(subject, body, config):
    """send email"""
    msg = MIMEMultipart()
    msg['From'] = config["sender_email"]
    msg['To'] = config["receiver_email"]
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    try:
        logging.info(f"connecting to SMTP server {config['server']}:{config['port']}...")
        server = smtplib.SMTP(config["server"], config["port"])
        server.starttls() # enable secure transmission mode
        server.login(config["sender_email"], config["password"])
        logging.info("SMTP server login success.")
        
        text = msg.as_string()
        server.sendmail(config["sender_email"], config["receiver_email"], text)
        logging.info(f"email sent to {config['receiver_email']}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logging.error(f"email send failed: SMTP authentication error. please check sender email and password/app password. error: {e}")
        return False
    except ConnectionRefusedError:
        logging.error(f"email send failed: connection refused. please check SMTP server address and port.")
        return False
    except Exception as e:
        logging.error(f"email send failed: unknown error. error: {e}")
        raise e
    finally:
        if 'server' in locals() and server:
            server.quit()
            logging.info("SMTP connection closed.")

def run_additional_cmds():
    '''run additional cmds'''
    for cmd in ADDITIONAL_CMDS:
        try:
        
            logging.info(f"run: {cmd}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                logging.info(f"run cmd success")
            else:
                logging.error(f"run cmd failed (Exitcode: {result.returncode}): {cmd}")
                if result.stderr.strip():
                    logging.error(f"ERR: {result.stderr.strip()}")
                if result.stdout.strip():
                    logging.info(f"OUT: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            logging.error(f"Time out: {cmd}")
        except Exception as e:
            logging.error(e)

def main():
    """main function"""
    logging.info("startup notifier script started...")
    
    # check if running with root permission, because journalctl and /var/log usually need root permission
    if os.geteuid() != 0:
        logging.warning("script not running with root permission, may not be able to read complete system log or write log file.")
        
    system_info = get_system_info()
    
    hostname = run_command("hostname") or "unknown host"
    subject = f"Learndl: host startup notification: {hostname}"
    
    body = f"""
hello,

your host '{hostname}' was started at:
{system_info['boot_time']}

--- last run status ---
last shutdown time: {system_info['last_shutdown_time']}
last shutdown status: {system_info['last_shutdown_status']}

--- status analysis ---
{system_info['abnormal_shutdown_reason']}

this email is sent by startup notifier script automatically.
"""
    
    logging.info("system info collected, preparing to send email.")
    email_success = send_email(subject, body.strip(), SMTP_CONFIG)
    if email_success:
        logging.info("Send mail success , run additional cmd")
        run_additional_cmds()
    else:
        logging.error("Send mail failed!")
    logging.info("script executed.")

if __name__ == "__main__":
    main() 