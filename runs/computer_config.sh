#!/bin/bash

# Computer Configuration File for learndl project
# This file contains configurations for different computers

# Function to get short computer name (before first dot)
get_short_name() {
    FULL_HOSTNAME=$(hostname)
    echo "$FULL_HOSTNAME" | cut -d'.' -f1
}

# Function to get computer configuration
get_computer_config() {
    SHORT_HOSTNAME=$(get_short_name)
    
    case "$OSTYPE" in
        "linux-gnu"*)
            case "$SHORT_HOSTNAME" in
                "mengkjin-server")
                    echo "/home/mengkjin/workspace/learndl|/home/mengkjin/workspace/learndl/.venv/bin/python"
                    # echo "/home/mengkjin/workspace/learndl|python3.10"
                    ;;
            esac
            ;;
        "darwin"*)
            case "$SHORT_HOSTNAME" in
                "Mathews-Mac")
                    echo "/Users/mengkjin/workspace/learndl|/Users/mengkjin/workspace/learndl/.venv/bin/python"
                    ;;
            esac
            ;;
    esac
}

# Function to display current computer info
show_computer_info() {
    SHORT_HOSTNAME=$(get_short_name)
    
    echo "=== Computer Information ==="
    echo "Full Hostname: $(hostname)"
    echo "Short Hostname: $SHORT_HOSTNAME"
    echo "OS Type: $OSTYPE"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "User: $USER"
    echo "Home: $HOME"
    echo "=========================="
}

# Function to list all known computers
list_known_computers() {
    echo "=== Known Computer Configurations ==="
    echo "Linux Computers:"
    echo "  - mengkjin-server"
    echo ""
    echo "macOS Computers:"
    echo "  - Mathews-Mac"
    echo "================================"
}

# Export functions for use in other scripts
export -f get_short_name
export -f get_computer_config
export -f show_computer_info
export -f list_known_computers
if [ "$OSTYPE" = "linux-gnu" ]; then
    export PYTHONPATH="/home/mengkjin/workspace/learndl":$PYTHONPATH
elif [ "$OSTYPE" = "darwin" ]; then
    export PYTHONPATH="/Users/mengkjin/workspace/learndl":$PYTHONPATH
fi
