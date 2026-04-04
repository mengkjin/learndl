"""macOS Ghostty via LaunchServices: ``open -na Ghostty.app --args``."""

from __future__ import annotations

import shlex

from ...util.process import popen_detached

from .verify import GhosttyVerifier


class GhosttyOpener:
    @classmethod
    def run(cls, cwd: str, command: str) -> None:
        """
        Open a new Ghostty window running ``cd`` + ``command`` via ``/bin/sh -c``.

        Ghostty's CLI notes that on macOS the GUI must be started with ``open``;
        ``--args`` forwards flags (e.g. ``-e``) to the app.
        """
        if not GhosttyVerifier.available():
            raise RuntimeError("Ghostty.app is not installed")
        inner = f"cd {shlex.quote(cwd)} && {command}"
        popen_detached(
            [
                "open",
                "-na",
                "Ghostty.app",
                "--args",
                "-e",
                "/bin/sh",
                "-c",
                inner,
            ]
        )
