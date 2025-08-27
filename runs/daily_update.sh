#!/bin/bash

# Check OS type
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    BASE_PATH="/home/mengkjin/workspace/learndl"
    PYTHON_CMD="python3.10"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    BASE_PATH="/Users/mengkjin/workspace/learndl"
    PYTHON_CMD="/Users/mengkjin/workspace/learndl/.venv/bin/python"
fi

# Check if Python interpreter is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD interpreter not found"
    echo "Please ensure Python 3 is installed"
    exit 1
fi

# Check if directory exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Directory $BASE_PATH does not exist"
    echo "Please check the working directory path"
    exit 1
fi

# Switch to working directory
cd "$BASE_PATH"

$PYTHON_CMD src/scripts/1_autorun/0_daily_update.py --source=bash --email=1
