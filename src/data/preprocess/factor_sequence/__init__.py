"""Factor-sequence preprocessing: explicit schemas, causal transforms, immutable bundles."""
from .schema import BuildConfig, KEYS, SHARED_KEYS
from .bundle import build, load, audit
