#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-04-12
# description: Update Context Library
# content: Scans context docs and outputs a targeted update brief. Run after source changes or when a doc mistake is found. Does not rewrite docs itself — outputs what to re-read and which STALE markers were found.
# email: True
# mode: shell
# parameters:
#   mode :
#       type : [changed, git, doc, stale, all]
#       desc : "changed=check specific files, git=auto-detect via git diff, doc=force-refresh one doc, stale=scan STALE markers, all=refresh everything"
#       required : True
#   target :
#       type : str
#       desc : "changed mode: comma-separated changed file paths (e.g. src/data/handler.py,src/data/block.py); doc mode: path to context doc (e.g. context/modules/data_pipeline.md)"
#       required : False
#   since_commit :
#       type : str
#       desc : "git mode only: commit ref to diff against (default HEAD~1)"
#       default : HEAD~1
#       required : False


"""
Modes:
  # Source-driven: which docs need refreshing based on changed files?
  python scripts/update_context.py --changed src/data/handler.py src/data/block.py

  # Git-driven: auto-detect changed files since last doc update
  python scripts/update_context.py --since-commit abc1234

  # Correction-driven: force-refresh one specific doc
  python scripts/update_context.py --doc context/modules/data_pipeline.md

  # Scan for flagged corrections in all docs
  python scripts/update_context.py --stale

  # Refresh everything
  python scripts/update_context.py --all


The correction flagging workflow is now:
  1. During a session, if you (or Claude) notice a wrong API in a doc, add <!-- STALE: what's wrong --> inline
   in the markdown
  2. Run scripts/0_check/2_update_context.py --mode stale (or via the app's dropdown) to get a brief of
  everything that needs fixing
  3. Claude reads the brief and re-explores the relevant sources to rewrite the flagged sections
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.proj.util import ScriptTool

ROOT = Path(__file__).resolve().parents[2]      # project root
INDEX_PATH = ROOT / 'context' / '_index.yaml'
STALE_RE = re.compile(r'<!--\s*STALE\s*:\s*(.*?)\s*-->', re.IGNORECASE)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_index() -> list[dict[str, Any]]:
    with open(INDEX_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f) or []

def files_to_affected_docs(changed: list[str], index: list[dict]) -> list[tuple[dict, list[str]]]:
    """Return (entry, matched_files) for each doc whose sources: overlap the changed list."""
    results = []
    for entry in index:
        sources = [s.rstrip('/') for s in entry.get('sources', [])]
        matched = [
            cf for cf in changed
            if any(cf == s or cf.startswith(s + '/') or s in cf for s in sources)
        ]
        if matched:
            results.append((entry, matched))
    return results

def scan_stale(doc_path: Path) -> list[tuple[int, str]]:
    """Return (line_number, reason) for every <!-- STALE: ... --> marker in a doc."""
    markers = []
    try:
        for i, line in enumerate(doc_path.read_text(encoding='utf-8').splitlines(), 1):
            for m in STALE_RE.finditer(line):
                markers.append((i, m.group(1).strip()))
    except Exception:
        pass
    return markers

def print_brief(entry: dict, reason: str,
                changed: list[str] | None = None,
                stale: list[tuple[int, str]] | None = None) -> None:
    doc      = entry['doc']
    sources  = entry.get('sources', [])
    sep      = '=' * 60
    print(f'\n{sep}')
    print(f'DOC    : {doc}')
    print(f'REASON : {reason}')
    if changed:
        print('CHANGED FILES:')
        for f in changed:
            print(f'  - {f}')
    print('SOURCE PATHS TO RE-READ:')
    for s in sources:
        print(f'  - {s}')
    if stale:
        print('STALE MARKERS:')
        for lineno, note in stale:
            print(f'  - line {lineno}: {note}')
    print(f'ACTION : Re-explore the source paths above and rewrite the relevant')
    print(f'         sections of {doc}. Add or remove <!-- STALE --> markers as needed.')

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@ScriptTool('update_context')
def main(mode: str | None = None,
         target: str | None = None,
         since_commit: str = 'HEAD~1',
         **kwargs) -> None:

    assert mode is not None, 'mode is required'
    index = load_index()

    # ── changed ──────────────────────────────────────────────────────────
    if mode == 'changed':
        assert target, "mode='changed' requires target (comma-separated file paths)"
        changed = [f.strip() for f in re.split(r'[,;]', target) if f.strip()]
        affected = files_to_affected_docs(changed, index)

        print(f'\n=== Context Update Brief  ·  mode: changed ===')
        print(f'Input files: {changed}')
        if not affected:
            print('No context docs are affected by these files.')
            return
        for entry, matched in affected:
            print_brief(entry, 'source files changed', matched, scan_stale(ROOT / entry['doc']))

    # ── git ──────────────────────────────────────────────────────────────
    elif mode == 'git':
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', since_commit, 'HEAD'],
                capture_output=True, text=True, cwd=ROOT, check=True
            )
            changed = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        except subprocess.CalledProcessError as e:
            print(f'git diff failed: {e.stderr.strip()}')
            return

        print(f'\n=== Context Update Brief  ·  mode: git (since {since_commit}) ===')
        if not changed:
            print(f'No changed files found since {since_commit}.')
            return
        print(f'Changed files ({len(changed)}):')
        for f in changed:
            print(f'  {f}')

        affected = files_to_affected_docs(changed, index)
        if not affected:
            print('No context docs are affected by these changes.')
            return
        for entry, matched in affected:
            print_brief(entry, f'source files changed since {since_commit}', matched,
                        scan_stale(ROOT / entry['doc']))

    # ── doc ──────────────────────────────────────────────────────────────
    elif mode == 'doc':
        assert target, "mode='doc' requires target (path to a context doc)"
        target_norm = target.replace('\\', '/')
        matching = [e for e in index if e['doc'].replace('\\', '/') == target_norm]

        print(f'\n=== Context Update Brief  ·  mode: doc ===')
        if not matching:
            print(f'WARNING: {target} not found in _index.yaml — outputting partial brief.')
            stale = scan_stale(ROOT / target)
            print(f'DOC    : {target}')
            print(f'REASON : Forced refresh requested')
            if stale:
                print('STALE MARKERS:')
                for lineno, note in stale:
                    print(f'  - line {lineno}: {note}')
            print(f'ACTION : Re-explore source code and rewrite {target}.')
            print(f'         Consider adding this doc to context/_index.yaml.')
            return
        for entry in matching:
            print_brief(entry, 'forced refresh requested',
                        stale=scan_stale(ROOT / entry['doc']))

    # ── stale ─────────────────────────────────────────────────────────────
    elif mode == 'stale':
        print(f'\n=== Context Update Brief  ·  mode: stale ===')
        found = False
        for entry in index:
            markers = scan_stale(ROOT / entry['doc'])
            if markers:
                found = True
                print_brief(entry, 'STALE markers found', stale=markers)
        if not found:
            print('No <!-- STALE: ... --> markers found in any context doc. All clear.')

    # ── all ───────────────────────────────────────────────────────────────
    elif mode == 'all':
        print(f'\n=== Context Update Brief  ·  mode: all ===')
        for entry in index:
            print_brief(entry, 'full refresh requested',
                        stale=scan_stale(ROOT / entry['doc']) or None)

    else:
        print(f"Unknown mode '{mode}'. Choose from: changed, git, doc, stale, all")


if __name__ == '__main__':
    main()
