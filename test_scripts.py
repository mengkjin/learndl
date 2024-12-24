import psutil
import fnmatch
from pathlib import Path

def get_running_scripts(script_type = ['*.py', '*.sh']):
    running_scripts : list[Path] = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline: continue
            for line in cmdline:
                if any(fnmatch.fnmatch(line, pattern) for pattern in script_type) and Path(line) != Path(__file__):
                    running_scripts.append(Path(line))
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return running_scripts

def main():
    running_scripts = get_running_scripts()
    if running_scripts:
        print(f"Suspension aborted due to running scripts: {running_scripts}\n")
    else:
        print("No matching scripts are running.")

if __name__ == '__main__':
    main()

