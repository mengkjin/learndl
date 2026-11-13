"""Opt-in research artifact delivery, independent of SMTP and training imports."""
from .core import ArtifactSpec, Config, publish_artifacts, receive

__all__ = ['ArtifactSpec', 'Config', 'publish_artifacts', 'receive']
