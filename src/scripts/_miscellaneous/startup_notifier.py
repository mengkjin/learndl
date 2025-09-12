#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        return False
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