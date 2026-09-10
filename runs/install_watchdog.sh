#!/bin/bash
set -euo pipefail

VERIFY_ONLY=0
REMOVE_LEGACY_CRON=0
for INSTALL_ARG in "$@"; do
    case "$INSTALL_ARG" in
        --verify) VERIFY_ONLY=1 ;;
        --remove-legacy-cron) REMOVE_LEGACY_CRON=1 ;;
        --help)
            echo "Usage: bash runs/install_watchdog.sh [--verify] [--remove-legacy-cron]"
            exit 0
            ;;
        *)
            echo "Unknown option: $INSTALL_ARG" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/systemd"
export PYTHONPATH="${PYTHONPATH:-}"

if [ -f "$SCRIPT_DIR/computer_config.sh" ]; then
    source "$SCRIPT_DIR/computer_config.sh"
fi

PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
if command -v get_computer_config >/dev/null 2>&1; then
    CONFIG="$(get_computer_config)"
    CONFIG_ROOT="$(echo "$CONFIG" | cut -d'|' -f1)"
    CONFIG_PYTHON="$(echo "$CONFIG" | cut -d'|' -f2)"
    [ -n "$CONFIG_ROOT" ] && PROJECT_ROOT="$CONFIG_ROOT"
    [ -n "$CONFIG_PYTHON" ] && PYTHON_PATH="$CONFIG_PYTHON"
fi

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Project directory not found: $PROJECT_ROOT" >&2
    exit 1
fi
if [ ! -x "$PYTHON_PATH" ]; then
    echo "Python interpreter is not executable: $PYTHON_PATH" >&2
    exit 1
fi

if [ "$VERIFY_ONLY" -eq 1 ]; then
    "$PYTHON_PATH" -m src.api.task_monitor.watchdog --help
    systemctl status learndl-watchdog.timer --no-pager || true
    echo "Watchdog verification completed."
    exit 0
fi

SERVICE_USER="$(id -un)"
SERVICE_GROUP="$(id -gn)"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
WATCH_UNITS="${LEARNDL_WATCHDOG_UNITS:-}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

sed \
    -e "s|@LEARNDL_USER@|$SERVICE_USER|g" \
    -e "s|@LEARNDL_GROUP@|$SERVICE_GROUP|g" \
    -e "s|@LEARNDL_HOME@|$SERVICE_HOME|g" \
    -e "s|@LEARNDL_ROOT@|$PROJECT_ROOT|g" \
    -e "s|@LEARNDL_PYTHON@|$PYTHON_PATH|g" \
    -e "s|@LEARNDL_WATCHDOG_UNITS@|$WATCH_UNITS|g" \
    "$TEMPLATE_DIR/learndl-watchdog.service.in" > "$TEMP_DIR/learndl-watchdog.service"

sudo install -m 0644 "$TEMP_DIR/learndl-watchdog.service" /etc/systemd/system/learndl-watchdog.service
sudo install -m 0644 "$TEMPLATE_DIR/learndl-watchdog.timer" /etc/systemd/system/learndl-watchdog.timer
sudo systemctl daemon-reload
sudo systemctl enable --now learndl-watchdog.timer
sudo systemctl start learndl-watchdog.service

if [ "$REMOVE_LEGACY_CRON" -eq 1 ]; then
    CURRENT_CRONTAB="$(crontab -l 2>/dev/null || true)"
    if [[ "$CURRENT_CRONTAB" == *task_monitor_maintenance.sh* ]]; then
        printf '%s\n' "$CURRENT_CRONTAB" | sed '/task_monitor_maintenance\.sh/d' | crontab -
        echo "Removed legacy task_monitor_maintenance.sh cron entries for $(id -un)."
    else
        echo "No legacy task monitor cron entry was found for $(id -un)."
    fi
fi

echo "Learndl watchdog installed."
echo "Timer status: systemctl status learndl-watchdog.timer"
echo "Logs: journalctl -u learndl-watchdog.service --since today --no-pager"
