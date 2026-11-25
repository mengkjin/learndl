"""Operator permissions shared by the loopback services and phone gateway."""

ROLE_HEADER = 'X-LearnDL-Role'
FORWARDED_HEADERS = ('X-Forwarded-For', 'X-Forwarded-Host', 'X-Forwarded-Proto')
GUEST_SERVICES = frozenset(('tasks', 'storage', 'compare', 'mail_archive', 'model_preview', 'experiments'))
ADMIN_REQUIRED = 'This operation requires an administrator.'


def is_guest(headers):
    # These loopback servers trust local operators. A forwarded request can
    # only narrow that trust, even if an outdated gateway omits the role.
    return any(name in headers for name in (ROLE_HEADER, *FORWARDED_HEADERS))


def role_page(page, headers):
    return page.replace('__MONITOR_ROLE__', 'guest' if is_guest(headers) else 'admin')


def permits_guest_action(service, path, data):
    action = data.get('action')
    if service == 'console':
        if path == '/api/restart-services':
            targets = data.get('targets')
            return isinstance(targets, list) and all(
                isinstance(target, dict) and isinstance(target.get('id'), str)
                and target['id'] in GUEST_SERVICES for target in targets)
        if path in ('/api/preferences', '/api/restart'):
            return True
        target = data.get('service')
        return (path == '/api/action' and action in ('start', 'stop', 'restart', 'scan')
                and isinstance(target, str) and target in GUEST_SERVICES)
    if service == 'tasks':
        return path in ('/api/remote/reload', '/api/kill/prepare', '/api/kill/confirm')
    if path != '/api/action':
        return False
    if service == 'mail_archive':
        return action in ('archive', 'unarchive', 'restore', 'open_attachment', 'folder')
    if service == 'model_preview':
        return action in ('start', 'stop', 'open_file', 'folder', 'model_folder')
    return service == 'experiments' and action in ('start', 'stop', 'stop_all', 'limit', 'email')
