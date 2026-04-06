"""macOS Ghostty via LaunchServices: ``open -na Ghostty.app --args``."""

from __future__ import annotations

import shlex

from ...util.process import popen_detached
from ...util.basic import BasicOpener
from .verify import GhosttyVerifier


class GhosttyOpener(BasicOpener):
    def available(self) -> bool:
        return GhosttyVerifier.available()

    def run(self, command: str, * , cwd: str | None = None, **kwargs) -> None:
        """
        Open a new Ghostty window running ``cd`` + ``command`` via ``/bin/sh -c``.

        Ghostty's CLI notes that on macOS the GUI must be started with ``open``;
        ``--args`` forwards flags (e.g. ``-e``) to the app.
        """
        assert self._available , f"{self.__class__.__name__} is not available"
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        popen_detached(
            [
                "open",
                "-na",
                "Ghostty.app",
                "--args",
                "-e",
                "/bin/sh",
                "-c",
                f'{command}\n',
            ]
        )
