"""Run callback microbenchmarks; add --short-tests for four gated training runs.

Example: uv run python scripts/5_test/benchmark_nn_callbacks.py --device cuda --short-tests
Default: all five candidates, nine alternating blocks, email JSON/CSV/log/evidence.
Use --only summary_latest basic_test or --no-email for focused local tests.
Retry mail without training: python -m src.proj.util.web.task_mail <output>/mail
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.res.model.diagnostics.callback_benchmark import main

if __name__=='__main__':main()
