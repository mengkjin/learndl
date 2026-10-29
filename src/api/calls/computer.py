"""
Direct calls related to computer operations of this project.
"""
from __future__ import annotations

import subprocess
from datetime import datetime

from src.proj import MACHINE , Logger , PATH
from src.proj.core import StrEnum
from src.api.util.direct_call import DirectCall

__all__ = ['SuspendThisMachine' , 'ChangePowerMode' , 'PrintDiskSpaceInfo', 'ProjectStorageReport', 'BrowseStorageReport']


class ProjectStorageReport(DirectCall):
    """Scan project disk usage, save an offline HTML report, and email it by default."""

    category = 'Computer'

    def run(self):
        from pathlib import Path
        from src.proj.util.cli.prompts import prompt_text
        from src.proj.util.filesys.storage_report import generate_report, format_size

        root = self.kwargs.get('root')
        output = self.kwargs.get('output')
        if root is None:
            entered = prompt_text(f'Scan root [Enter = {PATH.production}]')
            if entered is None:
                return None
            root = entered.strip() or PATH.production
            if output is None:
                entered = prompt_text('HTML output path [Enter = results/storage_reports/<timestamp>.html]')
                if entered is None:
                    return None
                output = entered.strip() or None
        root = Path(root).expanduser().resolve(strict=True)
        if output is None:
            output = root / 'results' / 'storage_reports' / f'storage_{datetime.now():%Y%m%d_%H%M%S_%f}.html'
        else:
            output = Path(output).expanduser()
            if not output.is_absolute():
                output = root / output
        if output.exists():
            raise FileExistsError(f'Report already exists: {output}')
        Logger.stdout(f'Scanning project storage: {root}')
        scan, report = generate_report(root, output, progress=Logger.stdout)
        Logger.stdout(f'Storage report saved: {report}')
        sent = None
        if self.kwargs.get('email', True):
            from src.proj.util.web.emailer import Email
            body = (f'Root: {root}\nAllocated space: {format_size(scan.node.size)}\n'
                    f'Status: {"complete" if scan.node.complete else "INCOMPLETE"}\n'
                    f'Errors: {scan.error_count}\nScan: {scan.started} — {scan.finished}\n'
                    f'Local report: {report}\nOpen the attached HTML in a browser.')
            try:
                sent = Email.send(
                    f'Project Storage Report {scan.finished}', body,
                    recipient=self.kwargs.get('recipient'), attachments=[report],
                    project_attachments=False, confirmation_message='Project Storage Report',
                )
            except Exception as exc:
                sent = False
                Logger.error(f'Report saved, but email failed: {exc}')
            if sent:
                Logger.stdout('Storage report email sent successfully.')
            else:
                Logger.alert1('Report saved locally; email was not sent. Check email logs for retry or configuration status.')
        return {'report': report, 'scan': scan, 'email_sent': sent}


class BrowseStorageReport(DirectCall):
    """Browse a saved report's directory snapshot on its original computer; Ctrl+C stops the service."""

    category = 'Computer'

    def run(self):
        from pathlib import Path
        from src.proj.util.cli.prompts import prompt_text
        from src.proj.util.filesys.storage_browser import StorageReportServer

        report = self.kwargs.get('report')
        if report is None:
            report = prompt_text('HTML report path (relative to production project, or absolute)')
            if report is None or not report.strip():
                return None
        report = Path(report).expanduser()
        if not report.is_absolute():
            report = PATH.production / report
        with StorageReportServer(report) as server:
            Logger.stdout(f'Browse storage report: {server.url}\nPress Ctrl+C to stop.')
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                Logger.stdout('Storage report browser stopped.')

class SuspendThisMachine(DirectCall):
    """Suspend the machine if not windows and no running python scripts"""
    category = 'Computer'
    def run(self) -> None:
        import psutil
        import fnmatch
        running_scripts = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline: 
                    continue
                for line in cmdline:
                    if any(fnmatch.fnmatch(line, pattern) for pattern in ['*.py' , '*.sh']) and line != __file__:
                        running_scripts.append(line)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if running_scripts:
            Logger.critical(f'Suspension aborted due to running scripts: {set(running_scripts)}')
        elif MACHINE.is_windows:
            Logger.critical(f'Suspension aborted due Windows platform')
        elif MACHINE.is_macos:
            Logger.critical(f'Suspension aborted due MacOS platform')
        elif MACHINE.is_linux:
            Logger.critical(f'Suspension denied! Please manually suspend the machine.')
        else:
            Logger.critical(f'Suspension applied')
            subprocess.run(['systemctl', 'suspend'])

class _PowerMode(StrEnum):
    BALANCED = 'balanced'
    POWER_SAVER = 'power-saver'
    PERFORMANCE = 'performance'
    NONE = 'none'

class ChangePowerMode(DirectCall):
    """
    Set Linux power profile via ``powerprofilesctl``; no-op log on Windows.
    Accepted modes:
    - 'balanced'
    - 'power-saver'
    - 'performance'
    - 'none'
    """
    category = 'Computer'
    def __init__(self , mode : _PowerMode | str | None = None , **kwargs):
        self.kwargs = kwargs | {'mode': _PowerMode(mode or 'none')}
    @property
    def mode(self) -> _PowerMode:
        return _PowerMode(self.kwargs['mode'])
    @classmethod
    def get_description(cls , mode : _PowerMode | str | None = None , **kwargs) -> str:
        mode = _PowerMode(mode or 'none')
        if mode == 'none':
            return 'Power mode is not set, skip the operation.'
        return f'Set Linux power profile to {mode} via ``powerprofilesctl``; no-op log on Windows.'
    def run(self) -> None:
        if self.mode is None:
            Logger.critical(f'Power mode is not set')
            return
        main_str = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Power set to {self.mode}'
        if MACHINE.is_windows:
            main_str += f' aborted due windows platform\n'
        else:
            main_str += f' applied\n'
            subprocess.run(['powerprofilesctl', 'set', self.mode])
        Logger.stdout(main_str , end = '')

class PrintDiskSpaceInfo(DirectCall):
    """Human-readable disk usage summary for the project root."""
    category = 'Computer'
    
    def run(self):
        import shutil , os
        def format_bytes(bytes_num):
            """Format byte count with binary SI steps (1024)."""
            suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            index = 0
            while bytes_num >= 1024 and index < len(suffixes) - 1:
                bytes_num /= 1024.0
                index += 1
            return f"{bytes_num:.2f} {suffixes[index]}"

        def get_disk_space_info():
            """Return dict of formatted total/used/free and usage percentages for ``PATH.main``."""
            total, used, free = shutil.disk_usage(PATH.main)
            percent_used = (used / total * 100) if total > 0 else 0
            percent_free = 100 - percent_used
            result = {
                'path': os.path.abspath(PATH.main),
                'total': format_bytes(total),
                'used': format_bytes(used),
                'free': format_bytes(free),
                'pct_used': round(percent_used, 2),
                'pct_free': round(percent_free, 2),
            }
            return result

        info = get_disk_space_info()
        Logger.stdout_pairs(info , title = 'Disk Space Info:')
        return info
