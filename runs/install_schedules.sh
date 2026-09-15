#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="${PYTHONPATH:-}"
if [ -f "$SCRIPT_DIR/computer_config.sh" ]; then
    source "$SCRIPT_DIR/computer_config.sh"
    CONFIG="$(get_computer_config)"
    if [ -n "$CONFIG" ]; then
        PROJECT_ROOT="${CONFIG%%|*}"
        PYTHON_PATH="${CONFIG#*|}"
    fi
fi
cd "$PROJECT_ROOT"
exec "$PYTHON_PATH" -m src.api.task_monitor.scheduling.installer "$@"
