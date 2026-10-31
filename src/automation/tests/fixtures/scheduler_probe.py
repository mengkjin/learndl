"""Harmless runner smoke-test payload; deliberately contains no project imports."""
import os

print(f'probe pid={os.getpid()} run={os.environ.get("LEARNDL_MANAGED_RUN")}', flush=True)
