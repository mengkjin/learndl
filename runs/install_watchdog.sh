#!/bin/bash
# Compatibility entrypoint for the unified scheduling installer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    --verify) shift; exec bash "$SCRIPT_DIR/install_schedules.sh" status "$@" ;;
    --remove-legacy-cron)
        echo 'Legacy cron is preserved. Comment out the old entry, then run install_schedules.sh apply.' >&2
        exit 2 ;;
    --help) exec bash "$SCRIPT_DIR/install_schedules.sh" --help ;;
    '') exec bash "$SCRIPT_DIR/install_schedules.sh" apply ;;
    *) exec bash "$SCRIPT_DIR/install_schedules.sh" "$@" ;;
esac
