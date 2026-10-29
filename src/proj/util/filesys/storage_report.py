"""Read-only allocated-space scan and standalone HTML reporting (standard library only)."""
from __future__ import annotations

import html
import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any


@dataclass
class StorageNode:
    path: str
    size: int = 0
    own: int = 0
    complete: bool = True
    target: str | None = None
    is_dir: bool = False
    children: list[StorageNode] = field(default_factory=list)


@dataclass
class StorageScan:
    root: Path
    node: StorageNode
    started: str
    finished: str
    entries: int
    disk_total: int
    disk_free: int
    errors: list[str]
    error_count: int


def display_depth(parts: tuple[str, ...]) -> int:
    """Depth relative to the scan root; unknown project categories retain two levels."""
    if parts and parts[0] == 'data':
        return 4 if len(parts) > 1 and parts[1] in {'DataBase', 'Interim', 'Export'} else 3
    if parts and parts[0] in {'models', 'results', '.logs', '.local_resources'}:
        return 3
    return 2


def format_size(value: int) -> str:
    number = float(value)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
        if number < 1024 or unit == 'PiB':
            return f'{number:.2f} {unit}'
        number /= 1024
    raise AssertionError('unreachable')


def scan_storage(root: str | Path, progress: Callable[[str], None | Any] | None = None) -> StorageScan:
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not hasattr(root.stat(), 'st_blocks'):
        raise RuntimeError('This platform does not expose allocated disk blocks (st_blocks).')
    started = datetime.now().astimezone().isoformat(timespec='seconds')
    errors: list[str] = []
    error_count = entries = 0
    seen: set[tuple[int, int]] = set()
    last_progress = time.monotonic()

    def error(node: StorageNode, exc: OSError) -> None:
        nonlocal error_count
        node.complete = False
        error_count += 1
        if len(errors) < 100:
            errors.append(f'{node.path}: {exc}')

    def walk(path: Path, parts: tuple[str, ...], visible: bool) -> StorageNode:
        nonlocal entries, last_progress
        node = StorageNode('/'.join(parts) or '.')
        entries += 1
        if progress and time.monotonic() - last_progress >= 2:
            progress(f'Scanned {entries:,} entries: {node.path}')
            last_progress = time.monotonic()
        try:
            info = path.lstat()
            is_dir = stat.S_ISDIR(info.st_mode)
            node.is_dir = is_dir
            key = (info.st_dev, info.st_ino)
            if not is_dir and info.st_nlink > 1:
                if key in seen:
                    return node
                seen.add(key)
            node.size = node.own = info.st_blocks * 512
            if stat.S_ISLNK(info.st_mode):
                node.target = os.readlink(path)
            if is_dir:
                with os.scandir(path) as iterator:
                    names = sorted(entry.name for entry in iterator)
                for name in names:
                    child_parts = (*parts, name)
                    retain = visible and len(child_parts) <= display_depth(child_parts)
                    child = walk(path / name, child_parts, retain)
                    node.size += child.size
                    node.complete = node.complete and child.complete
                    # Only directories and links are report rows; ordinary files are aggregated.
                    if retain and (child.is_dir or child.target is not None or not child.complete):
                        node.children.append(child)
                    else:
                        node.own += child.size
        except OSError as exc:
            error(node, exc)
        return node

    node = walk(root, (), True)
    disk = shutil.disk_usage(root)
    if progress:
        progress(f'Scan complete: {entries:,} entries, {format_size(node.size)}, {error_count} errors')
    return StorageScan(root, node, started, datetime.now().astimezone().isoformat(timespec='seconds'),
                       entries, disk.total, disk.free, errors, error_count)


def render_html(scan: StorageScan) -> str:
    escape = html.escape

    def row(node: StorageNode, parent_size: int, depth: int) -> str:
        percentage = f'{node.size / parent_size:.1%}' if parent_size else '—'
        label = escape(node.path.rsplit('/', 1)[-1])
        warning = '' if node.complete else ' <strong>统计不完整</strong>'
        link = f' → {escape(node.target)}' if node.target is not None else ''
        children = ''.join(row(child, node.size, depth + 1)
                           for child in sorted(node.children, key=lambda child: (-child.size, child.path)))
        own = (f'<p class="own">本层占用：{format_size(node.own)}'
               '（目录自身及未单列内容）</p>')
        return (f'<details class="node" data-path="{escape(node.path, quote=True)}" '
                f'data-size="{node.size}" {"open" if depth == 0 else ""}>'
                f'<summary><span>{label}{link}{warning}</span><b>{format_size(node.size)}</b>'
                f'<small>{percentage}</small></summary>{own}<div class="children">{children}</div></details>')

    errors = ''.join(f'<li>{escape(error)}</li>' for error in scan.errors)
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>项目磁盘占用报告</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fb;color:#17243b}
header,.tree{background:white;padding:24px;border-radius:12px;margin-bottom:20px}h1{margin-top:0}
summary{cursor:pointer;padding:10px 4px;border-bottom:1px solid #e5e9f0;display:flex;gap:16px}
summary:before{content:'▸'}details[open]>summary:before{content:'▾'}summary span{flex:1;overflow-wrap:anywhere}
summary b{white-space:nowrap}summary small{width:64px;text-align:right}.children{margin-left:22px}
.own{color:#667085;font-size:12px;margin:8px 24px}strong{color:#b54708}input,select,button{padding:9px;margin:5px}
input{min-width:240px}li{overflow-wrap:anywhere}[hidden]{display:none!important}
</style><header><h1>项目磁盘占用报告</h1>''' + f'''
<p>根目录：{escape(str(scan.root))}</p>
<p>项目占用：<b>{format_size(scan.node.size)}</b> · 状态：{'完整' if scan.node.complete else '统计不完整'} · 扫描条目：{scan.entries:,}</p>
<p>根目录所在文件系统：总容量 {format_size(scan.disk_total)} · 剩余 {format_size(scan.disk_free)}</p>
<p>开始：{escape(scan.started)} · 完成：{escape(scan.finished)}</p>
<p>按分配磁盘块统计，包含目录自身；父子合计不能重复相加。软链接不跟随；硬链接仅计入稳定遍历顺序遇到的首个路径。
显示层级以下的内容全部计入合计。扫描期间文件可能变化；跨文件系统的挂载目录也会扫描，剩余空间仅对应根目录所在文件系统。</p>
<label>搜索路径 <input id="search" type="search" placeholder="例如 DataBase 或 gru"></label>
<label>同级排序 <select id="sort"><option value="size">占用降序</option><option value="name">路径名称</option></select></label>
<button id="expand">全部展开</button><button id="collapse">全部折叠</button></header>
<main class="tree">{row(scan.node, scan.node.size, 0)}</main>
<footer><h2>扫描错误：{scan.error_count}</h2><p>最多展示前 100 条错误。</p><ul>{errors}</ul></footer>
''' + '''<script>
const nodes=[...document.querySelectorAll('details.node')];
document.querySelector('#search').addEventListener('input',event=>{
 const q=event.target.value.toLowerCase();
 for(const node of [...nodes].reverse()){
  const match=node.dataset.path.toLowerCase().includes(q);
  const descendant=[...node.querySelector('.children').children].some(child=>!child.hidden);
  node.hidden=!!q&&!match&&!descendant;
  node.open=q ? !node.hidden : node===nodes[0];
 }
});
document.querySelector('#sort').addEventListener('change',event=>{
 for(const container of document.querySelectorAll('.children')){
  const children=[...container.children];
  children.sort((a,b)=>event.target.value==='size' ? Number(b.dataset.size)-Number(a.dataset.size)||a.dataset.path.localeCompare(b.dataset.path) : a.dataset.path.localeCompare(b.dataset.path));
  container.append(...children);
 }
});
document.querySelector('#expand').onclick=()=>nodes.forEach(node=>node.open=true);
document.querySelector('#collapse').onclick=()=>nodes.forEach(node=>node.open=node===nodes[0]);
</script></html>'''


def write_report(scan: StorageScan, output: str | Path) -> Path:
    """Create a report only after the scan, without replacing existing reports."""
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(render_html(scan))
    return output
