"""Keep the original 0.7.2 patch import and preflight command working."""
from src.automation.version_autofix import load_patch_support


_support = load_patch_support('0.7.2', 'result_layout')
run = _support.run

if __name__ == '__main__':
    _support.main()
