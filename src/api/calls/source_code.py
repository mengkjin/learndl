"""
Direct calls related to source code management of this project.
1. git management
2. check code issues
3. check dependency version
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable , cast

from src.proj import MACHINE , PATH , Logger , Base
from src.proj.util.functional.ask import AskFor
from src.api.util.direct_call import DirectCall

__all__ = [
    'GitCommitSync' , 'GitClearPull' , 
    'RunRuffPyrightCheck' , 'CheckCodeIssues' ,
    'CheckDependencyVersion' ,
]

# %% project code related operations
class GitCommitSync(DirectCall):
    """Commit and sync the code. Args: verbose_level: 0: no output, 1: only success, 2: all output"""
    category = 'Codes'
    def __init__(self , commit_message : str = 'auto commit' , verbose_level : Base.lit._0_3 = 1 , **kwargs):
        self.kwargs = kwargs | {'commit_message': commit_message, 'verbose_level': verbose_level}
    @property
    def commit_message(self) -> str:
        return self.kwargs['commit_message']
    @property
    def verbose_level(self) -> Base.lit._0_3:
        return self.kwargs['verbose_level']
    def run(self) -> None:
        prefixes = f'{MACHINE.name} >> {datetime.now().strftime('%Y%m%d')} : {self.commit_message}'
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', prefixes], check=True)
        subprocess.run(['git', 'pull', '--rebase'], check=True)
        subprocess.run(['git', 'push'], check=True)
        if self.verbose_level >= 1:
            Logger.success(f"Finish commit and sync")
        if self.verbose_level >= 2:
            Logger.stdout(f"Commit message: {prefixes}")

class GitClearPull(DirectCall):
    """Clear local changes and pull latest code. Args: verbose_level: 0: no output, 1: only success, 2: all output"""
    category = 'Codes'
    def __init__(self , verbose_level : Base.lit._0_3 = 1 , **kwargs):
        self.kwargs = kwargs | {'verbose_level': verbose_level}
    @property
    def verbose_level(self) -> Base.lit._0_3:
        verbose_level = self.kwargs['verbose_level']
        return cast(Base.lit._0_3, verbose_level)
    def run(self) -> None:
        import shutil
        assert not MACHINE.platform_coding, "Git Pull is not available on coding platform"

        # clean local changes
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
        subprocess.run(['git', 'clean', '-fd'], check=True)
        if self.verbose_level >= 1:
            Logger.success(f"Clean local changes done")
        
        # pull latest code
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True)
        if self.verbose_level >= 1:
            Logger.success(f"Pull latest code done")
        if self.verbose_level >= 2:
            Logger.stdout(result.stdout)
        
        empty_folders = []
        for folder in [*PATH.main.joinpath('src').rglob('*/') , *PATH.main.joinpath('configs').rglob('*/')][::-1]:
            if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
                subfiles = [x for x in folder.rglob('*') if x.is_file()]
                if not len(subfiles):
                    if self.verbose_level >= 2:
                        Logger.stdout(f"Removing empty folder: {folder}" , indent = 1)
                    folder.rmdir()
                    empty_folders.append(folder)
                elif all([x.suffix == '.pyc' for x in subfiles]):
                    if self.verbose_level >= 2:
                        Logger.stdout(f"Removing folder with only pyc files: {folder}" , indent = 1)
                    shutil.rmtree(folder)
                    empty_folders.append(folder)
                else:
                    Logger.error(f"Error removing folder: {folder}:")
                    Logger.error(f"Subfiles: {subfiles}")
        if self.verbose_level >= 1:
            Logger.success(f"Removed {len(empty_folders)} empty folders done")

class RunRuffPyrightCheck(DirectCall):
    """Run ruff pyright check on the project code. """
    category = 'Codes'
    @classmethod
    def check_args(cls , check_name : str) -> list[str]:
        if check_name == 'ruff':
            return ['ruff' , 'check' , 'src' , 'scripts']
        elif check_name == 'pyright':
            return ['pyright' , 'src' , 'scripts']
        else:
            raise ValueError(f"Invalid check name: {check_name}")
    def run(self) -> None:
        for check_name in ['ruff' , 'pyright']:
            Logger.note(f"Running {check_name} check on the project code...")
            ret = subprocess.run(self.check_args(check_name) , capture_output=True, text=True, check=False)
            Logger.success(f"{check_name.title()} check passed.") if ret.returncode == 0 else Logger.alert2(f"{check_name.title()} check failed: ")
            if ret.stdout.strip():
                Logger.stdout(ret.stdout.strip())
            if ret.stderr.strip():
                Logger.stderr(ret.stderr.strip())

class CheckCodeIssues(DirectCall):
    """Check the code issues in the project code."""
    category = 'Codes'
    @staticmethod
    def _skip_path(path : Path , exempt_paths : list[Path] , exempt_names : list[str]) -> bool:
        if not path.is_file() or path.name in exempt_names or any(path.is_relative_to(exempt_path) for exempt_path in exempt_paths):
            return True
        return False
    @classmethod
    def _check_future_annotations_existence(cls ,path : Path) -> bool:
        """Check future annotations existence."""
        if cls._skip_path(path , [] , ['__init__.py' , '__version__.py']):
            return True
        return 'from __future__ import annotations' in path.read_text()
    @classmethod
    def _check___all___existence(cls ,path : Path) -> bool:
        """Check __all__ existence."""
        exempt_paths = [
            PATH.main.joinpath('src' , 'func') ,
            PATH.main.joinpath('src' , 'proj' , 'core' , 'types') ,
        ]
        if cls._skip_path(path , exempt_paths , ['__init__.py' , '__version__.py']):
            return True
        return '__all__' in path.read_text()
    @classmethod
    def _check_module_docstring_existence(cls , path : Path) -> bool:
        """Check module docstring existence."""
        if cls._skip_path(path , [] , ['__init__.py' , '__version__.py']):
            return True
        for line in path.read_text().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            return line.startswith('"""') or line.startswith("'''")
        return False

    @classmethod
    def _run_check(cls , checker : Callable[[Path], bool]) -> None:
        """Run the check."""
        Logger.note(f"{checker.__doc__}...")
        files = [file for file in PATH.main.joinpath('src').rglob('*.py') if not checker(file)]
        if files:
            [Logger.stdout(file , indent = 1) for file in files]
        else:
            Logger.success(f'No files found with the issue {checker.__doc__}.')

    @classmethod
    def _menu_for_checking(cls):
        checkers : list[Callable[[Path], bool]] = [getattr(cls, name) for name in dir(cls) if name.startswith('_check_')]
        options = ['All checks in one' , 'Run ruff pyright check' , *[checkers.__doc__ for checkers in checkers]]
        flag = AskFor.Options(options , confirm = False , multiple = False , title = f'What code issues to check?')
        if flag.result is None:
            return flag
        selection = options.index(flag.result)
        if selection == 0:
            for checker in checkers:
                cls._run_check(checker)
            RunRuffPyrightCheck.go()
        elif selection == 1:
            RunRuffPyrightCheck.go()
        else:
            checker = checkers[selection - 2]
            cls._run_check(checker)
        return flag

    def run(self) -> None:
        for loop_flag in AskFor.LoopTillExit(False , message = f'Do you want to check more code issues?' , max_trials = 100):
            flag = self._menu_for_checking()
            loop_flag.set_flag(flag)

class CheckDependencyVersion(DirectCall):
    """Check the dependency version in the project code if they are newer than the ones in pyproject.toml."""
    category = 'Codes'

    @classmethod
    def get_dependencies_from_toml(cls) -> dict[str, str]:
        """Parses pyproject.toml and extracts dependencies with a '>=' constraint."""
        import tomllib , re

        toml_path = PATH.main.joinpath('pyproject.toml')
        if not toml_path.exists():
            raise FileNotFoundError(f"Error: {toml_path} not found.")

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        # Standard PEP 621 dependencies location
        dependencies = data.get("project", {}).get("dependencies", [])
        
        # Regex to capture package name and the '>=' version boundary
        # Example match: "requests>=2.28.0" -> name="requests", version="2.28.0"
        ge_pattern = re.compile(r"^([a-zA-Z0-9_\-\[\]]+)>=\s*([0-9a-zA-Z\.\-\+]+)")
        
        target_deps = {}
        for dep in dependencies:
            # Strip out environment markers or extras notation for clean parsing
            clean_dep = dep.split(";")[0].strip()
            match = ge_pattern.match(clean_dep)
            if match:
                pkg_name, min_version = match.groups()
                # Normalize package name (lowercase, dashes to underscores for imports)
                normalized_name = pkg_name.split("[")[0].lower().replace("-", "_")
                target_deps[normalized_name] = min_version

        return target_deps
    
    def check_dependency_version(self) -> None:
        import importlib.metadata
        from packaging.version import Version

        dependencies = self.get_dependencies_from_toml()

        if not dependencies:
            Logger.success("No dependencies with '>=' constraints found in pyproject.toml.")
            return

        Logger.note(f"Scanning {len(dependencies)} packages with '>=' constraints...\n")
        Logger.stdout(f"{'Package':<25} | {'pyproject.toml (>=)':<20} | {'Installed Version':<20}")
        Logger.stdout("-" * 71)

        found_newer = False
        for pkg_name, constraint_version in dependencies.items():
            try:
                # Use importlib.metadata to grab the exact installed version safely
                # Fall back to an active runtime import logic if metadata fails
                try:
                    installed_version_str = importlib.metadata.version(pkg_name)
                except importlib.metadata.PackageNotFoundError:
                    # Direct import fallback
                    module = importlib.import_module(pkg_name)
                    installed_version_str = getattr(module, "__version__", None)
                    if not installed_version_str:
                        raise ImportError

                # Compare using packaging.version to handle complex semver tags correctly
                if Version(installed_version_str) > Version(constraint_version):
                    Logger.stdout(f"{pkg_name:<25} | {constraint_version:<20} | {installed_version_str:<20} (Newer!)")
                    found_newer = True

            except (ImportError, ValueError):
                # Handles if the package isn't installed in the environment or has bad semver
                Logger.alert2(f"{pkg_name:<25} | {constraint_version:<20} | [Not installed / Unreadable]")

        if not found_newer:
            Logger.success("\nAll installed packages perfectly match their lower '>=' boundaries.")

    def run(self) -> None:
        for _ in AskFor.LoopTillExit(True , message = f'Do you want to check more dependency version?' , max_trials = 100):
            self.check_dependency_version()