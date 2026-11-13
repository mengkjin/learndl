"""python -m src.automation.artifact_exchange --help"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from .core import ArtifactSpec, Config, DEFAULT_CONFIG, atomic_json, publish_artifacts, receive, retry, sandboxed
from .management import migration, service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    init.add_argument('--role', choices=['publisher', 'receiver'], required=True)
    init.add_argument('--enable', action='store_true')
    init.add_argument('--source-host', default=None)
    pub = commands.add_parser('publish')
    pub.add_argument('source', type=Path)
    pub.add_argument('--task', required=True)
    pub.add_argument('--run-id', required=True)
    pub.add_argument('--kind', choices=['reports', 'tensorboard', 'optuna'], default='reports')
    pub.add_argument('--category', default='custom')
    pub.add_argument('--name', default='')
    pub.add_argument('--experiment', required=True)
    pub.add_argument('--destination', default='')
    pub.add_argument('--status', default='success')
    rec = commands.add_parser('receive')
    rec.add_argument('--once', action='store_true', required=True)
    for name in ('retry', 'status', 'doctor'):
        commands.add_parser(name)
    m = commands.add_parser('migrate-results')
    m.add_argument('action', choices=['plan', 'apply', 'rollback'])
    s = commands.add_parser('service')
    s.add_argument('action', choices=['plan', 'install', 'status', 'uninstall'])
    args = parser.parse_args(argv)
    if sandboxed():
        print(json.dumps({'status': 'skipped', 'reason': 'autofix'}))
        return 0
    config = Config.load(args.config)
    if args.command == 'init':
        if args.config.exists():
            raise FileExistsError(f'Edit existing configuration explicitly: {args.config}')
        config = Config(role=args.role, enabled=args.enable)
        if args.source_host:
            config.source_host = args.source_host
        atomic_json(args.config, {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()})
        output = {'config': str(args.config), 'enabled': config.enabled, 'role': config.role}
    elif args.command == 'publish':
        output = publish_artifacts(args.task, args.run_id, [ArtifactSpec(args.source, args.kind, args.category, args.name, experiment=args.experiment, destination=args.destination)], args.status, config=config)
    elif args.command == 'receive':
        config.state.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger('artifact_exchange')
        handler = RotatingFileHandler(config.state / 'receiver.log', maxBytes=2_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            output = receive(config)
            logger.info(json.dumps(output, ensure_ascii=False))
        except BlockingIOError:
            output = {'status': 'busy'}
        except Exception:
            logger.exception('Receiver failed')
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()
    elif args.command == 'retry':
        output = retry(config)
    elif args.command == 'migrate-results':
        output = migration(args.action, config)
    elif args.command == 'service':
        output = service(args.action, args.config, config)
    else:
        output = {'enabled': config.enabled, 'role': config.role, 'shared_root': str(config.shared_root),
                  'folder_marker': (config.shared_root / '.stfolder').exists(),
                  'pending_packages': sum(1 for _ in config.inbox.rglob('*.transfer.zip')) if config.inbox.exists() else 0,
                  'outbox': [json.loads(p.read_text()) for p in (config.state / 'outbox').glob('*/delivery.json')],
                  'last_receive': json.loads((config.state / 'last_receive.json').read_text()) if (config.state / 'last_receive.json').exists() else []}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return int(isinstance(output, list) and any(r.get('status') == 'failed' for r in output))


if __name__ == '__main__':
    sys.exit(main())
