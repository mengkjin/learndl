"""Project-relative path resolution and autocomplete for CLI prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys

from src.proj import PATH
from src.proj.util.cli.magic import magic_autocomplete_tokens

__all__ = [
    'EXCLUDED_DIRNAMES',
    'MagicOrProjectPathCompleter',
    'ProjectPathCompleter',
    'build_path_prompt_key_bindings',
    'is_magic_input',
    'resolve_project_path',
]

EXCLUDED_DIRNAMES: frozenset[str] = frozenset({
    '.git',
    '__pycache__',
    '.venv',
    'node_modules',
    '.logs',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
})

_PROJECT_ROOTS: tuple[Path, ...] = (PATH.main, PATH.production)
_RESOLVE_ROOT_ORDER: tuple[Path, ...] = (PATH.main, PATH.production)


def _unique_roots(roots: tuple[Path, ...] = _PROJECT_ROOTS) -> tuple[Path, ...]:
    """Deduplicate project roots by resolved path (main and production are often identical)."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return tuple(unique)


def is_magic_input(text: str) -> bool:
    """Return True when *text* looks like a magic command prefix (``/help``, etc.)."""
    if not text.startswith('/'):
        return False
    return any(token.startswith(text) for token in magic_autocomplete_tokens())


def build_path_prompt_key_bindings() -> KeyBindings:
    """Key bindings for project path input: ``/`` drills down, Enter accepts completion."""
    bindings = KeyBindings()

    @bindings.add(Keys.ControlM, eager=True)
    def accept_completion_or_submit(event: KeyPressEvent) -> None:
        buff = event.current_buffer
        if buff.complete_state is not None:
            completion = buff.complete_state.current_completion
            if completion is not None:
                buff.apply_completion(completion)
            buff.complete_state = None
        else:
            buff.append_to_history()
            event.app.exit(result=buff.text)

    @bindings.add('/', eager=True)
    def insert_slash_and_complete(event: KeyPressEvent) -> None:
        buff = event.current_buffer
        text_before = buff.document.text_before_cursor
        if buff.complete_state is not None:
            buff.complete_state = None
        if is_magic_input(text_before + '/'):
            buff.insert_text('/')
            return
        if not text_before.endswith('/'):
            buff.insert_text('/')
        buff.start_completion(select_first=False)

    return bindings


def _is_under_project(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in _PROJECT_ROOTS)


def resolve_project_path(raw: str) -> Path:
    """Resolve a project-relative path, preferring ``PATH.main`` over ``PATH.production``.

    Args:
        raw: Project-relative path string from the user.

    Returns:
        Resolved absolute path.

    Raises:
        FileNotFoundError: Path does not exist under either project root.
        ValueError: Path is absolute or escapes project roots.
    """
    cleaned = raw.strip().replace('\\', '/')
    if cleaned.startswith('/') or (len(cleaned) > 1 and cleaned[1] == ':'):
        raise ValueError('Path must be relative to the project (not absolute)')
    cleaned = cleaned.lstrip('./')
    if not cleaned:
        raise ValueError('Path must not be empty')

    for root in _RESOLVE_ROOT_ORDER:
        candidate = (root / cleaned).resolve()
        if candidate.exists():
            if not _is_under_project(candidate):
                raise ValueError(f'Path is outside the project: {raw}')
            return candidate

    raise FileNotFoundError(f'Path not found under project roots: {raw}')


def _relative_display(path: Path) -> str:
    return str(PATH.relative(path)).replace('\\', '/')


def _iter_completion_entries(directory: Path) -> Iterable[tuple[str, bool]]:
    """Yield ``(name, is_dir)`` entries for completion, skipping excluded dirs."""
    if not directory.is_dir():
        return
    try:
        names = sorted(directory.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return
    for entry in names:
        if entry.name in EXCLUDED_DIRNAMES or entry.name.startswith('__'):
            continue
        yield entry.name, entry.is_dir()


class MagicOrProjectPathCompleter(Completer):
    """Delegate to magic or project path completion based on current input."""

    def __init__(self) -> None:
        from src.proj.util.cli.magic import MagicCommandCompleter

        self._path = ProjectPathCompleter()
        self._magic = MagicCommandCompleter()

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if is_magic_input(text):
            yield from self._magic.get_completions(document, complete_event)
        else:
            yield from self._path.get_completions(document, complete_event)


class ProjectPathCompleter(Completer):
    """Filesystem completer rooted at project ``main`` and ``production`` trees."""

    def __init__(self, roots: tuple[Path, ...] = _PROJECT_ROOTS) -> None:
        self._roots = _unique_roots(tuple(root.resolve() for root in roots))

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor.replace('\\', '/')
        if is_magic_input(text):
            return
        if not text:
            yield from self._complete_top_level(name_prefix='')
            return

        root, remainder = self._match_root(text)
        if root is None:
            yield from self._complete_top_level(name_prefix=text)
            return

        partial = remainder.replace('\\', '/')
        if not partial:
            yield from self._complete_directory_entries(root, name_prefix='')
            return

        if partial.endswith('/'):
            directory = (root / partial).resolve()
            if directory.is_dir() and _is_under_project(directory):
                yield from self._complete_directory_entries(directory, name_prefix='')
            return

        parent_rel, _, name_prefix = partial.rpartition('/')
        directory = (root / parent_rel).resolve() if parent_rel else root
        if not directory.is_dir() or not _is_under_project(directory):
            return

        yield from self._complete_directory_entries(directory, name_prefix=name_prefix)

    def _complete_top_level(self, *, name_prefix: str) -> Iterable[Completion]:
        """List immediate children of unique project roots (e.g. src/, models/)."""
        seen_names: set[str] = set()
        for root in self._roots:
            for name, is_dir in _iter_completion_entries(root):
                if name_prefix and not name.startswith(name_prefix):
                    continue
                if name in seen_names:
                    continue
                seen_names.add(name)
                completion = name if not is_dir else f'{name}/'
                yield Completion(completion, start_position=-len(name_prefix))

    def _complete_directory_entries(
        self,
        directory: Path,
        *,
        name_prefix: str,
    ) -> Iterable[Completion]:
        """Yield suffix-only completions for entries under *directory*."""
        if not directory.is_dir():
            return
        for name, is_dir in _iter_completion_entries(directory):
            if name_prefix and not name.startswith(name_prefix):
                continue
            completion = name if not is_dir else f'{name}/'
            yield Completion(completion, start_position=-len(name_prefix))

    def _match_root(self, text: str) -> tuple[Path | None, str]:
        normalized = text.replace('\\', '/')
        best: tuple[Path | None, str] | None = None
        for root in self._roots:
            rel = _relative_display(root).replace('\\', '/')
            rel_prefix = '' if rel == '.' else rel + '/'
            if normalized == rel or normalized.startswith(rel_prefix):
                remainder = normalized[len(rel_prefix):] if rel_prefix else normalized
                if best is None or len(rel_prefix) > len(best[1]):
                    best = (root, remainder)
            elif rel_prefix.startswith(normalized) or rel.startswith(normalized):
                if best is None or len(rel) > len(best[1]):
                    best = (root, '')
        return best if best is not None else (None, normalized)
