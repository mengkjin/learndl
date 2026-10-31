"""Compatibility entrypoint for src.automation.scheduling.install_flow; no implementation belongs here."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    if not __package__:
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    runpy.run_module('src.automation.scheduling.install_flow', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('src.automation.scheduling.install_flow')
