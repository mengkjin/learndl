"""Immutable, size-bounded MIME packages. No network or project configuration access."""
from __future__ import annotations

import copy
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import uuid
import zipfile
from email import policy
from email.message import EmailMessage, Message
from pathlib import Path

MAX_BYTES = 30_000_000
MAX_ATTACHMENT_BYTES = 20_000_000
BODY_BYTES = 1_000_000


def wire(message: Message) -> bytes:
    """The bytes measured here are also passed to SMTP.sendmail."""
    # Freeze multipart boundaries once, just as the stdlib serializer does.
    for part in message.walk():
        if part.is_multipart() and not part.get_boundary():
            part.set_boundary('learndl-' + uuid.uuid4().hex)
    result = copy.deepcopy(message)
    for name in ('Bcc', 'Resent-Bcc'):
        if name in result:
            del result[name]
    return result.as_bytes(policy=policy.SMTP)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'))


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def fsync_dir(path: Path) -> None:
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decoded(part: Message) -> bytes | None:
    # email.get_payload(decode=True) splits base64 into millions of line objects.
    # binascii (via b64decode) accepts whitespace directly, keeping old 150 MiB
    # outbox entries within the watchdog's 1 GiB memory limit.
    if str(part.get('Content-Transfer-Encoding', '')).lower() == 'base64' and not part.is_multipart():
        payload = part.get_payload()
        return base64.b64decode(payload.encode('ascii') if isinstance(payload, str) else payload)
    return part.get_payload(decode=True)


def fingerprint(message: Message, *, normalize_body_newlines: bool = False) -> str:
    """Ignore randomly generated MIME boundaries, dates and Message-IDs on retries."""
    def item(part):
        headers = [(k.lower(), str(policy.default.header_factory(k, str(v)))) for k, v in part.items()
                   if k.lower() not in {'content-type', 'mime-version', 'date', 'message-id'}]
        header = policy.default.header_factory('Content-Type', str(part.get('Content-Type', 'text/plain')))
        params = [part.get_content_type(), {k: v for k, v in header.params.items() if k != 'boundary'}]
        if part.is_multipart():
            payload = [item(p) for p in part.get_payload()]
        else:
            data = decoded(part) or b''
            # SMTP and local MIME serialization use CRLF and LF respectively.
            # Only inline text is equivalent under this conversion; attachment
            # bytes must remain exact. Keep default fingerprints/receipt IDs stable.
            if (normalize_body_newlines and part.get_content_maintype() == 'text'
                    and not part.get_filename() and part.get_content_disposition() != 'attachment'):
                data = data.replace(b'\r\n', b'\n')
            payload = digest(data)
        return headers, params, payload
    return digest(json.dumps(item(message), ensure_ascii=False, sort_keys=True).encode())


def attachment_capacity(budget: int) -> int:
    """Largest raw attachment fitting the Base64/CRLF budget, capped at 20 MB."""
    low, high = 0, MAX_ATTACHMENT_BYTES
    while low < high:
        middle = (low + high + 1) // 2
        encoded = 4 * ((middle + 2) // 3)
        if encoded + 2 * ((encoded + 75) // 76) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def balanced_sizes(size: int, cap: int) -> list[int]:
    """Minimum number of volumes, differing by at most one byte."""
    count = max(1, (size + cap - 1) // cap)
    quotient, remainder = divmod(size, count)
    return [quotient + (index < remainder) for index in range(count)]


def needs_preparation(message: Message, limit: int = MAX_BYTES) -> bool:
    return len(wire(message)) > limit or any(
        (bool(p.get_filename()) and len(decoded(p)) > MAX_ATTACHMENT_BYTES)
        or (p.get_content_maintype() == 'text' and not p.get_filename()
            and len(p.as_bytes(policy=policy.SMTP)) > BODY_BYTES)
        for p in message.walk() if not p.is_multipart())


def _safe_name(name: str, used: set[str]) -> str:
    name = name.replace('\\', '/').split('/')[-1]
    name = ''.join(c for c in name if c >= ' ' and c != '\x7f').strip('.')[:160] or 'attachment'
    base, number = name, 1
    while name.casefold() in used:
        number += 1
        name = f'{number}-{base}'
    used.add(name.casefold())
    return name


def _contents(message: Message):
    """Preserve multipart/related as MHTML so cid resources stay intact."""
    files, bodies, used = [], [], set()
    def visit(part):
        if part.get_content_type() == 'multipart/related':
            files.append((_safe_name('正文.mhtml', used), wire(part)))
        elif part.get_filename() or part.get_content_disposition() == 'attachment':
            data = decoded(part)
            if data is None:  # attached message/rfc822 and other container attachments
                data = part.as_bytes(policy=policy.SMTP)
            files.append((_safe_name(str(part.get_filename() or 'attachment.eml'), used), data))
        elif part.is_multipart():
            if part.get_content_type() not in ('multipart/mixed', 'multipart/alternative'):
                raise ValueError(f'Unsupported MIME container: {part.get_content_type()}')
            for child in part.get_payload():
                visit(child)
        elif part.get_content_maintype() == 'text':
            raw = decoded(part) or b''
            text = raw.decode(part.get_content_charset() or 'utf-8', errors='replace')
            if part.get_content_subtype() == 'plain' and len(part.as_bytes(policy=policy.SMTP)) <= BODY_BYTES:
                bodies.append(text)
            else:
                ext = 'html' if part.get_content_subtype() == 'html' else 'txt'
                files.append((_safe_name(f'正文.{ext}', used),
                              (('<meta charset="utf-8">' + text) if ext == 'html' else text).encode('utf-8')))
                if ext == 'txt':
                    bodies.append(text[:2000] + '\n[完整正文见附件]')
                else:
                    bodies.append('[完整 HTML 正文见附件；内嵌资源邮件保存为 MHTML]')
        else:
            raise ValueError(f'Unsupported inline MIME part: {part.get_content_type()}')
    visit(message)
    body = '\n\n'.join(bodies)
    if len(body.encode()) > min(BODY_BYTES, 100_000):
        files.append((_safe_name('完整正文.txt', used), body.encode()))
        body = body[:2000] + '\n[完整正文见附件]'
    return body, files


def prepare(message: Message, directory: Path, batch_id: str, limit: int = MAX_BYTES, *, source_size: int | None = None) -> dict:
    """Create payloads + manifest in a caller-owned staging directory; publish together."""
    if limit < 16_384 or limit > MAX_BYTES:
        raise ValueError(f'Mail size limit must be between 16384 and {MAX_BYTES}')
    directory.mkdir(parents=True, exist_ok=True)
    subject = str(message.get('Subject', ''))
    batch_key = digest(batch_id.encode())[:20]
    manifest = dict(schema_version=2, batch_id=batch_id, fingerprint=fingerprint(message),
                    recipient=str(message.get('To', '')), subject=subject, max_mime_bytes=limit,
                    messages=[], attachments=[], transformed=False)
    if (source_size is None or source_size <= limit) and not needs_preparation(message, limit):
        msg = copy.deepcopy(message)
        if 'Message-ID' not in msg:
            msg['Message-ID'] = f'<{batch_key}.1@learndl.local>'
        messages = [wire(msg)]
        if len(messages[0]) > limit:
            messages = []
    else:
        messages = []
    if not messages:
        manifest['transformed'] = True
        body, files = _contents(message)
        note = (f'\n\n批次: {batch_id}\n请保留全部编号邮件的附件。ZIP 可直接解压；'
                '7z 分卷须放在同一目录，用 7-Zip / Keka 打开 .7z.001。'
                'MHTML 保存完整 HTML 与内嵌资源，可用支持 MHTML 的浏览器打开。\n')
        def build(items, index=999999, total=999999):
            msg = EmailMessage()
            for key in ('From', 'To', 'Cc', 'Reply-To'):
                if message.get(key):
                    msg[key] = str(message[key])
            msg['Subject'] = f'{subject} [{index}/{total}]'
            msg['Message-ID'] = f'<{batch_key}.{index}@learndl.local>'
            msg.set_content(body + note)
            for name, data in items:
                msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=name)
            return wire(msg)
        if len(build([])) > limit // 2:
            if body:
                files.insert(0, ('邮件正文.txt', body.encode()))
            body = '[正文较长，完整内容见附件。]'
        if len(build([])) >= limit:
            raise ValueError('Mail headers exceed the size budget')
        prepared = []
        for number, (name, data) in enumerate(files, 1):
            entry = {'name': name, 'size': len(data), 'sha256': digest(data)}
            manifest['attachments'].append(entry)
            if len(data) <= MAX_ATTACHMENT_BYTES and len(build([(name, data)])) <= limit:
                prepared.append((name, data))
                continue
            package_name = f'{batch_key}-{number:03d}'
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(name, data)
            zipped = buf.getvalue()
            if len(zipped) <= MAX_ATTACHMENT_BYTES and len(build([(package_name+'.zip', zipped)])) <= limit:
                prepared.append((package_name+'.zip', zipped))
                entry['package'] = package_name+'.zip'
                continue
            executable = shutil.which('7zz') or shutil.which('7z')
            if not executable:
                raise RuntimeError('7-Zip missing: install 7zz or 7z to split oversized attachments')
            work = directory / f'.compress-{number}'
            work.mkdir()
            (work / name).write_bytes(data)
            archive_path = work / (package_name+'.7z')
            try:
                # Compress once, then split the archive stream into balanced volumes.
                # 7-Zip/Keka reassemble .7z.001, .002, ... before extracting.
                result = subprocess.run([executable, 'a', '-t7z', '-mx=1',
                                         str(archive_path), '--', name],
                                        cwd=work, capture_output=True, timeout=300)
                if result.returncode:
                    raise RuntimeError(f'7-Zip failed ({result.returncode}): {result.stderr.decode(errors="replace")[:500]}')
                size = archive_path.stat().st_size
                volume_name = package_name+'.7z.001'
                overhead = len(build([(volume_name, b'')]))
                cap = attachment_capacity(limit - overhead)
                if cap < 1024:
                    raise ValueError('Cannot fit archive volumes in mail size budget')
                lengths = balanced_sizes(size, cap)
                parts = []
                with archive_path.open('rb') as archive:
                    for index, length in enumerate(lengths, 1):
                        part = (f'{package_name}.7z.{index:03d}', archive.read(length))
                        if len(build([part])) > limit:
                            raise ValueError('Archive volume exceeds mail size budget')
                        parts.append(part)
                prepared.extend(parts)
                entry['volumes'] = [dict(name=n, size=len(d), sha256=digest(d)) for n, d in parts]
            finally:
                shutil.rmtree(work)
        # Keep manifests small and independently usable even for many attachments.
        for number, entry in enumerate(manifest['attachments'], 1):
            data = json.dumps(entry, ensure_ascii=False, indent=2).encode()
            name = f'{batch_key}-{number:03d}-sha256.json'
            if len(build([(name, data)])) > limit:
                raise ValueError('Attachment manifest exceeds mail budget')
            prepared.append((name, data))
        groups = [[]]
        for attachment in prepared:
            if len(build(groups[-1] + [attachment])) > limit:
                groups.append([])
            groups[-1].append(attachment)
        messages = [build(group, index, len(groups)) for index, group in enumerate(groups, 1)]
    for index, raw in enumerate(messages, 1):
        if len(raw) > limit:
            raise ValueError('Final MIME exceeds configured size limit')
        name = f'message-{index:04d}.eml'
        atomic_bytes(directory / name, raw)
        manifest['messages'].append(dict(file=name, sha256=digest(raw), mime_bytes=len(raw),
                                         sent=False, attempts=0, next_retry_at=0, status='pending'))
    atomic_json(directory / 'manifest.json', manifest)
    return manifest
