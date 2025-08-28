#!/bin/bash

# Source computer configuration file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/computer_config.sh" ]; then
    source "$SCRIPT_DIR/computer_config.sh"
fi

# Get configuration using the function from computer_config.sh
if command -v get_computer_config >/dev/null 2>&1; then
    CONFIG=$(get_computer_config)
    BASE_PATH=$(echo "$CONFIG" | cut -d'|' -f1)
    PYTHON_CMD=$(echo "$CONFIG" | cut -d'|' -f2)
else
    # Fallback configuration if computer_config.sh is not available
    echo "Warning: computer_config.sh not found, using fallback configuration"
    BASE_PATH=$(dirname "$SCRIPT_DIR")
    PYTHON_CMD="python"
fi

cd "$BASE_PATH"
$PYTHON_CMD src/scripts/1_autorun/1_weekly_update.py --source=bash --email=1
