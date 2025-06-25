#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import logging
import os
import sys

# --- 配置部分 ---
# 请务必修改以下配置为你自己的信息
# 强烈建议使用环境变量或专门的配置文件来存储敏感信息，而不是直接硬编码在代码中
SMTP_CONFIG = {
    "server": os.environ.get("SMTP_SERVER", "smtp.example.com"),  # 你的 SMTP 服务器地址, 例如: smtp.gmail.com
    "port": int(os.environ.get("SMTP_PORT", 587)),                # 你的 SMTP 服务器端口 (通常 587 用于 TLS, 465 用于 SSL)
    "sender_email": os.environ.get("SMTP_SENDER", "sender@example.com"), # 发件人邮箱
    "receiver_email": os.environ.get("SMTP_RECEIVER", "receiver@example.com"), # 收件人邮箱
    "password": os.environ.get("SMTP_PASSWORD", "YOUR_APP_PASSWORD") # 发件人邮箱的密码或应用专用密码
}

# 要在邮件发送后运行的额外脚本路径 (支持 .py, .sh 等)
# 可以通过环境变量 ADDITIONAL_SCRIPT_PATH 设置，如果不设置则不运行额外脚本
ADDITIONAL_SCRIPT_PATH = os.environ.get("ADDITIONAL_SCRIPT_PATH", "")

# 日志文件路径
LOG_FILE = "/var/log/startup_notifier.log"

# --- 日志设置 ---
# 设置日志记录，方便调试
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # 同时输出到控制台
    ]
)

def run_command(command):
    """安全地执行 shell 命令并返回输出"""
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
        logging.error(f"命令 '{command}' 执行失败: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        logging.error(f"命令未找到: {command.split()[0]}. 请确保相关工具 (如 journalctl) 已安装。")
        return None

def get_system_info():
    """获取系统启动和上次关机信息"""
    info = {
        "boot_time": "未知",
        "last_shutdown_time": "未知",
        "last_shutdown_status": "未知",
        "abnormal_shutdown_reason": "无"
    }

    # 1. 获取本次启动时间
    boot_time_str = run_command("uptime -s")
    if boot_time_str:
        info["boot_time"] = boot_time_str

    # 2. 分析上次启动日志 (journalctl -b -1)
    previous_boot_log = run_command("journalctl -b -1 --no-pager")
    
    if not previous_boot_log:
        info["last_shutdown_status"] = "信息不足 (可能是首次启动或日志已轮替)"
        return info

    # 3. 获取上次关机时间 (上次会话的最后一条日志时间)
    # 使用 short-iso 格式可以获得带年份和时区的精确时间
    last_log_time_str = run_command("journalctl -b -1 -n 1 --no-pager -o short-iso")
    if last_log_time_str:
        # 格式如: 2023-10-27T15:30:00+0800, 我们清理一下格式
        info["last_shutdown_time"] = last_log_time_str.split('+')[0].replace('T', ' ')

    # 4. 判断上次关机是否正常
    if "Reached target Shutdown" in previous_boot_log or \
       "Reached target Power-Off" in previous_boot_log or \
       "Reached target Reboot" in previous_boot_log:
        info["last_shutdown_status"] = "正常关机"
    else:
        info["last_shutdown_status"] = "异常关机"
        # 5. 如果是异常关机，尝试分析原因
        # -p 3 表示 'err' 级别及更高（crit, alert, emerg）
        error_logs = run_command("journalctl -b -1 -p err --no-pager -n 10") # 获取最后10条错误日志
        if error_logs:
            info["abnormal_shutdown_reason"] = f"系统日志中发现错误，最后相关错误信息如下:\n---\n{error_logs}\n---"
        else:
            info["abnormal_shutdown_reason"] = "在上次启动日志中未找到明确的错误(error)信息。可能的原因包括：\n" \
                                           "  - 强制断电 (例如，直接拔掉电源)\n" \
                                           "  - 系统冻结后被强制重启\n" \
                                           "  - 内核崩溃 (Kernel Panic) 但未能记录到日志\n" \
                                           "  - 某些硬件故障"

    return info

def send_email(subject, body, config):
    """发送邮件，返回是否成功"""
    msg = MIMEMultipart()
    msg['From'] = config["sender_email"]
    msg['To'] = config["receiver_email"]
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    try:
        logging.info(f"正在连接到 SMTP 服务器 {config['server']}:{config['port']}...")
        server = smtplib.SMTP(config["server"], config["port"])
        server.starttls() # 启用安全传输模式
        server.login(config["sender_email"], config["password"])
        logging.info("SMTP 服务器登录成功。")
        
        text = msg.as_string()
        server.sendmail(config["sender_email"], config["receiver_email"], text)
        logging.info(f"邮件已成功发送至 {config['receiver_email']}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logging.error(f"邮件发送失败：SMTP 认证错误。请检查发件人邮箱和密码/应用专用密码。错误: {e}")
        return False
    except ConnectionRefusedError:
        logging.error(f"邮件发送失败：连接被拒绝。请检查 SMTP 服务器地址和端口。")
        return False
    except Exception as e:
        logging.error(f"邮件发送时发生未知错误: {e}")
        return False
    finally:
        if 'server' in locals() and server:
            server.quit()
            logging.info("SMTP 连接已关闭。")

def run_additional_script(script_path):
    """运行额外的脚本 (支持 .py, .sh 等)"""
    if not script_path:
        logging.info("未配置额外脚本路径，跳过额外脚本执行。")
        return
    
    if not os.path.exists(script_path):
        logging.error(f"额外脚本文件不存在: {script_path}")
        return
    
    if not os.access(script_path, os.R_OK):
        logging.error(f"无法读取额外脚本文件: {script_path}")
        return
    
    # 检查脚本是否有执行权限
    if not os.access(script_path, os.X_OK):
        logging.warning(f"脚本文件没有执行权限: {script_path}，尝试添加执行权限...")
        try:
            os.chmod(script_path, 0o755)
            logging.info(f"已为脚本添加执行权限: {script_path}")
        except Exception as e:
            logging.error(f"无法为脚本添加执行权限: {e}")
            return
    
    logging.info(f"开始执行额外脚本: {script_path}")
    
    try:
        # 根据文件扩展名决定执行方式
        _, ext = os.path.splitext(script_path)
        
        if ext.lower() == '.py':
            # Python 脚本使用 python 解释器
            cmd = [sys.executable, script_path]
        elif ext.lower() == '.sh':
            # Shell 脚本使用 bash 执行
            cmd = ['/bin/bash', script_path]
        else:
            # 其他脚本尝试直接执行
            cmd = [script_path]
        
        logging.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时，因为 shell 脚本可能需要更长时间
            cwd=os.path.dirname(script_path) if os.path.dirname(script_path) else None  # 在脚本所在目录执行
        )
        
        if result.returncode == 0:
            logging.info(f"额外脚本执行成功: {script_path}")
            if result.stdout.strip():
                logging.info(f"额外脚本输出: {result.stdout.strip()}")
        else:
            logging.error(f"额外脚本执行失败 (退出码: {result.returncode}): {script_path}")
            if result.stderr.strip():
                logging.error(f"额外脚本错误输出: {result.stderr.strip()}")
            if result.stdout.strip():
                logging.info(f"额外脚本标准输出: {result.stdout.strip()}")
                
    except subprocess.TimeoutExpired:
        logging.error(f"额外脚本执行超时 (超过10分钟): {script_path}")
    except Exception as e:
        logging.error(f"执行额外脚本时发生异常: {e}")

def main():
    """主函数"""
    logging.info("启动通知脚本开始执行...")
    
    # 检查是否以 root 权限运行，因为 journalctl 和 /var/log 通常需要
    if os.geteuid() != 0:
        logging.warning("脚本未使用 root 权限运行，可能无法读取完整的系统日志或写入日志文件。")
        
    system_info = get_system_info()
    
    hostname = run_command("hostname") or "未知主机"
    subject = f"主机启动通知: {hostname}"
    
    body = f"""
您好,

您的主机 '{hostname}' 已于以下时间启动：
{system_info['boot_time']}

--- 上次运行状态 ---
上次关机时间: {system_info['last_shutdown_time']}
上次关机状态: {system_info['last_shutdown_status']}

--- 状态分析 ---
{system_info['abnormal_shutdown_reason']}

此邮件由启动通知脚本自动发送。
"""
    
    logging.info("系统信息收集完毕，准备发送邮件。")
    email_success = send_email(subject, body.strip(), SMTP_CONFIG)
    
    if email_success:
        logging.info("邮件发送成功，准备执行额外脚本。")
        # 只有邮件发送成功后才运行额外脚本
        run_additional_script(ADDITIONAL_SCRIPT_PATH)
    else:
        logging.error("邮件发送失败，跳过额外脚本执行。")
    
    logging.info("脚本执行完毕。")

if __name__ == "__main__":
    main() 