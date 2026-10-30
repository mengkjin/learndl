"""Compatibility imports/CLI for existing RL mail records; implementation is shared."""
from src.proj.util.web.task_mail import TaskResultMail, retry_delivery, save_json, main

__all__ = ['TaskResultMail', 'retry_delivery', 'save_json']

if __name__ == '__main__':
    raise SystemExit(main())
