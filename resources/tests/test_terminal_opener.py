"""
Example: open Terminal, run test.py, then discover the child Python PID.

``Shell.run`` returns before the new terminal has started Python — use
:class:`ProcessDiscovery` polling. Matches are sorted by start time; ``[-1]`` is newest.
"""

from __future__ import annotations

import platform
import shutil
import tempfile
import time
from pathlib import Path

from src.proj.util.shell_opener import ProcessDiscovery, Shell

temp_dir = tempfile.mkdtemp()

def main() -> None:
    temp_dir = tempfile.mkdtemp()
    temp_script = Path(temp_dir) / "test.py"
    with open(temp_script , "w") as f:
        
        f.write("""
import time

def main():
    print('start to wait for 2 seconds')
    time.sleep(2)
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

    if platform.system() == "Darwin" and not shutil.which("cmux"):
        print("cmux not installed (optional)")

    Shell.run_python(temp_script , py_path = 'uv run' , option = 'cmux')
    hits = ProcessDiscovery.wait_for_running_instances(script=temp_script.resolve())
    if hits:
        print("found PIDs (oldest→newest):", hits)
        print("newest:", hits[-1])
    else:
        print(
            "no matching Python process within timeout — "
            "install psutil extra, check script path, or increase ProcessDiscovery._wait_timeout",
        )

    time.sleep(3)
    temp_script.unlink()
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
