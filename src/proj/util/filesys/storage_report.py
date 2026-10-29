"""Read-only allocated-space scan and standalone HTML reporting (standard library only)."""
from __future__ import annotations

import html
import os
import shutil
import stat
import time
import tempfile
import uuid
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
    node_id: int = 0
    has_children: bool = False


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
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)


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


def scan_storage(root: str | Path, progress: Callable[[str], None | Any] | None = None, *,
                 on_node: Callable[[StorageNode, int | None], None] | None = None,
                 exclude: Path | None = None) -> StorageScan:
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

    def walk(path: Path, parts: tuple[str, ...], visible: bool, parent_id: int | None = None) -> StorageNode:
        nonlocal entries, last_progress
        node = StorageNode('/'.join(parts) or '.')
        entries += 1
        node.node_id = entries
        if progress and time.monotonic() - last_progress >= 2:
            progress(f'Scanned {entries:,} entries: {node.path}')
            last_progress = time.monotonic()
        try:
            info = path.lstat()
            is_dir = stat.S_ISDIR(info.st_mode)
            node.is_dir = is_dir
            key = (info.st_dev, info.st_ino)
            if stat.S_ISLNK(info.st_mode):
                node.target = os.readlink(path)
            if not is_dir and info.st_nlink > 1:
                if key in seen:
                    if on_node:
                        on_node(node, parent_id)
                    return node
                seen.add(key)
            node.size = node.own = info.st_blocks * 512
            if is_dir:
                with os.scandir(path) as iterator:
                    names = sorted(entry.name for entry in iterator)
                for name in names:
                    if path / name == exclude:
                        continue
                    child_parts = (*parts, name)
                    retain = visible and len(child_parts) <= display_depth(child_parts)
                    child = walk(path / name, child_parts, retain, node.node_id)
                    node.size += child.size
                    node.complete = node.complete and child.complete
                    node.has_children = True
                    if retain:
                        node.children.append(child)
                    else:
                        node.own += child.size
        except OSError as exc:
            error(node, exc)
        if on_node:
            on_node(node, parent_id)
        return node

    node = walk(root, (), True)
    disk = shutil.disk_usage(root)
    if progress:
        progress(f'Scan complete: {entries:,} entries, {format_size(node.size)}, {error_count} errors')
    return StorageScan(root, node, started, datetime.now().astimezone().isoformat(timespec='seconds'),
                       entries, disk.total, disk.free, errors, error_count)


def render_node(node: StorageNode, parent_size: int, depth: int = 1) -> str:
    """Shared escaped rendering for snapshot rows and lazily fetched rows."""
    escape = html.escape
    percentage = f'{node.size / parent_size:.1%}' if parent_size else '—'
    label = escape(node.path.rsplit('/', 1)[-1])
    warning = '' if node.complete else ' <strong>统计不完整</strong>'
    link = f' → {escape(node.target)}' if node.target is not None else ''
    children = ''.join(render_node(child, node.size, depth + 1)
                       for child in sorted(node.children, key=lambda child: (-child.size, child.path)))
    button = ('<button class="load-next" type="button" hidden>加载下一层</button>'
              if node.is_dir and node.has_children and not node.children else '')
    if not node.is_dir:
        kind = '链接' if node.target is not None else '文件'
        return (f'<div class="node file" data-path="{escape(node.path, quote=True)}" '
                f'data-id="{node.node_id}" data-size="{node.size}">'
                f'<div class="entry-row"><span><small class="kind">{kind}</small> {label}{link}{warning}</span>'
                f'<b>{format_size(node.size)}</b><small>{percentage}</small></div></div>')
    return (f'<details class="node" data-path="{escape(node.path, quote=True)}" '
            f'data-id="{node.node_id}" data-size="{node.size}" {"open" if depth == 0 else ""}>'
            f'<summary><span>{label}{link}{warning}{button}</span><b>{format_size(node.size)}</b>'
            f'<small>{percentage}</small></summary><div class="children">{children}</div></details>')


def render_html(scan: StorageScan) -> str:
    escape = html.escape

    errors = ''.join(f'<li>{escape(error)}</li>' for error in scan.errors)
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>项目磁盘占用报告</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fb;color:#17243b}
header,.tree{background:white;padding:24px;border-radius:12px;margin-bottom:20px}h1{margin-top:0}
summary,.entry-row{padding:10px 4px;border-bottom:1px solid #e5e9f0;display:flex;gap:16px}
summary{cursor:pointer}summary:before{content:'▸'}details[open]>summary:before{content:'▾'}
summary span,.entry-row span{flex:1;overflow-wrap:anywhere}.entry-row:before{content:'·'}
summary b,.entry-row b{white-space:nowrap}summary small,.entry-row>small{width:64px;text-align:right}
.kind{color:#667085;font-size:11px}.children{margin-left:22px}
.load-next{font:inherit;font-size:12px;padding:3px 7px;margin:0 0 0 10px}strong{color:#b54708}input,select,button{padding:9px;margin:5px}
input[type="search"]{min-width:240px}li{overflow-wrap:anywhere}[hidden]{display:none!important}
</style><header><h1>项目磁盘占用报告</h1>''' + f'''
<meta name="storage-report-id" content="{scan.report_id}">
<p>根目录：{escape(str(scan.root))}</p>
<p>项目占用：<b>{format_size(scan.node.size)}</b> · 状态：{'完整' if scan.node.complete else '统计不完整'} · 扫描条目：{scan.entries:,}</p>
<p>根目录所在文件系统：总容量 {format_size(scan.disk_total)} · 剩余 {format_size(scan.disk_free)}</p>
<p>开始：{escape(scan.started)} · 完成：{escape(scan.finished)}</p>
<p>按分配磁盘块统计，包含目录自身；父子合计不能重复相加。软链接不跟随；硬链接仅计入稳定遍历顺序遇到的首个路径。
显示层级以下的内容全部计入合计。扫描期间文件可能变化；跨文件系统的挂载目录也会扫描，剩余空间仅对应根目录所在文件系统。</p>
<p id="browse-status">静态报告：交互运行 Project Storage Report 后选择本地打开，即可动态浏览文件和文件夹。附件不会连接本机服务。</p>
<label>搜索路径（已加载文件及文件夹） <input id="search" type="search" placeholder="例如 DataBase 或 gru"></label>
<label>同级排序 <select id="sort"><option value="size">占用降序</option><option value="name">路径名称</option></select></label>
<label><input id="hide-special" type="checkbox" checked>隐藏以 . / ~ 开头的文件和文件夹（总占用不变）</label>
<button id="expand">全部展开</button><button id="collapse">全部折叠</button></header>
<main class="tree">{render_node(scan.node, scan.node.size, 0)}</main>
<footer><h2>扫描错误：{scan.error_count}</h2><p>最多展示前 100 条错误。</p><ul>{errors}</ul></footer>
''' + '''<script>
const storageBase = null; // STORAGE_SESSION
const allNodes=()=>[...document.querySelectorAll('.node')];
function isSpecialPath(path){
 return path!=='.' && path.split('/').some(part=>part.startsWith('.')||part.startsWith('~'));
}
function applySearch(reset=false){
 const nodes=allNodes();
 const q=document.querySelector('#search').value.toLowerCase();
 const hideSpecial=document.querySelector('#hide-special').checked;
 for(const node of [...nodes].reverse()){
  const match=node.dataset.path.toLowerCase().includes(q);
  const descendant=[...(node.querySelector('.children')?.children||[])].some(child=>!child.hidden);
  node.hidden=(hideSpecial&&isSpecialPath(node.dataset.path)) || (!!q&&!match&&!descendant);
  if(node.tagName==='DETAILS'){
   if(q) node.open=!node.hidden;
   else if(reset) node.open=node===nodes[0];
  }
 }
}
function applySort(){
 const mode=document.querySelector('#sort').value;
 for(const container of document.querySelectorAll('.children')){
  const children=[...container.children];
  children.sort((a,b)=>mode==='size' ? Number(b.dataset.size)-Number(a.dataset.size)||a.dataset.path.localeCompare(b.dataset.path) : a.dataset.path.localeCompare(b.dataset.path));
  container.append(...children);
 }
}
document.querySelector('#search').oninput=()=>applySearch(true);
document.querySelector('#hide-special').onchange=()=>applySearch();
applySearch();
document.querySelector('#sort').onchange=applySort;
document.querySelector('#expand').onclick=()=>allNodes().filter(n=>n.tagName==='DETAILS').forEach(node=>node.open=true);
document.querySelector('#collapse').onclick=()=>allNodes().filter(n=>n.tagName==='DETAILS').forEach((node,i)=>node.open=i===0);
function enableLoading(){
 if(storageBase) document.querySelectorAll('.load-next').forEach(button=>button.hidden=false);
}
if(storageBase){
 document.querySelector('#browse-status').textContent='本机动态浏览：加载首次扫描的目录快照，不重新扫描磁盘。';
 enableLoading();
}
document.querySelector('.tree').addEventListener('click',async event=>{
 const button=event.target.closest('.load-next');
 if(!button) return;
 event.preventDefault(); event.stopPropagation();
 if(!storageBase||button.disabled) return;
 const node=button.closest('details.node');
 button.disabled=true; button.textContent='加载中…';
 const controller=new AbortController();
 const timeout=setTimeout(()=>controller.abort(),15000);
 try{
  const response=await fetch(storageBase+'children?id='+encodeURIComponent(node.dataset.id),
                             {signal:controller.signal,credentials:'omit'});
  if(!response.ok) throw new Error('HTTP '+response.status+': '+await response.text());
  const data=await response.json();
  node.querySelector('.children').innerHTML=data.html;
  node.open=true; button.remove(); enableLoading(); applySort(); applySearch();
 }catch(error){
  button.disabled=false;
  const reason=error.name==='AbortError' ? '请求超时' : error.message;
  button.textContent='加载失败，重试（'+reason+'）';
  button.title='请确认本机浏览服务仍在运行。'+error.message;
 }finally{
  clearTimeout(timeout);
 }
});
</script></html>'''


def write_report(scan: StorageScan, output: str | Path) -> Path:
    """Create a report only after the scan, without replacing existing reports."""
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(render_html(scan))
    return output


def generate_report(root: str | Path, output: str | Path,
                    progress: Callable[[str], Any] | None = None) -> tuple[StorageScan, Path]:
    """Stage a same-scan directory index and publish HTML last, without overwriting."""
    from .storage_browser import SnapshotIndex

    output = Path(output).expanduser().absolute()
    index = output.with_suffix('.sqlite3')
    if output == index:
        raise ValueError('Report output must not use the .sqlite3 extension')
    if output.exists() or index.exists():
        raise FileExistsError(f'Report or index already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.storage-stage-', dir=output.parent) as temporary:
        stage = Path(temporary).resolve()
        with SnapshotIndex(stage / 'index.sqlite3') as writer:
            scan = scan_storage(root, progress, on_node=writer.add, exclude=stage)
            content = render_html(scan)
            writer.finish(scan, content)
        staged_html = stage / 'report.html'
        staged_html.write_text(content, encoding='utf-8')
        # Atomic no-clobber publication; an interrupted run can leave an orphan index,
        # but never an HTML report referring to an unfinished index.
        os.link(stage / 'index.sqlite3', index)
        try:
            os.link(staged_html, output)
        except BaseException:
            index.unlink()
            raise
    return scan, output
