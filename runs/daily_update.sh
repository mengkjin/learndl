#!/bin/bash

CURRENT_SHELL=$(basename "$SHELL")
if [ "$CURRENT_SHELL" = "zsh" ] && [ -f ~/.zshrc ]; then
    echo "Loading ~/.zshrc for environment variables..."
    source ~/.zshrc
elif [ "$CURRENT_SHELL" = "bash" ] && [ -f ~/.bashrc ]; then
    echo "Loading ~/.bashrc for environment variables..."
    source ~/.bashrc
elif [ -f ~/.zshrc ]; then
    echo "Loading ~/.zshrc for environment variables (fallback)..."
    source ~/.zshrc
elif [ -f ~/.bashrc ]; then
    echo "Loading ~/.bashrc for environment variables (fallback)..."
    source ~/.bashrc
else
    echo "Warning: No shell configuration file found (~/.bashrc or ~/.zshrc)"
fi

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
$PYTHON_CMD src/scripts/1_autorun/0_daily_update.py --source=bash --email=1
