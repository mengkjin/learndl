#!/usr/bin/env bash
# Manual entry: preflight/build/audit, six sequential ordinary schedules, reports.
set -euo pipefail

factorseq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
factorseq_python="$factorseq_root/.venv/bin/python"
if [[ ! -x "$factorseq_python" ]]; then
    printf 'Project Python not found: %s\n' "$factorseq_python" >&2
    exit 1
fi

# Keep this entry to exactly six candidates with one seed. The research suite
# remains available via 6_factor_sequence_verify.py for larger experiments.
for factorseq_arg in "$@"; do
    case "$factorseq_arg" in
        --research|--research=*|--ablations|--ablations=*)
            printf 'This entry runs six models with one seed; research/ablations use 6_factor_sequence_verify.py.\n' >&2
            exit 2
            ;;
    esac
done

cd -- "$factorseq_root"
export PYTHONUNBUFFERED=1
exec "$factorseq_python" "$factorseq_root/scripts/4_train/6_factor_sequence_verify.py" "$@"
