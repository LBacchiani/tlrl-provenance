"""Version and source identity for replayable semantic machinery."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path


ENGINE_VERSION = "0.3.0"
SEMANTICS_VERSION = "future-ltlf-operational-provenance/1"


@lru_cache(maxsize=1)
def engine_source_digest() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        relative = path.name.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
