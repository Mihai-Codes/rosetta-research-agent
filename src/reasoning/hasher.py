"""Canonical SHA-256 hashing of reasoning traces.

The hash is the **trust anchor** of Rosetta Alpha. It must be:
1. **Deterministic** — re-running the same agent on the same input produces the
   same hash (modulo `timestamp`/`thesis_id`, which is why we exclude them
   below for content-only hashes).
2. **Cross-language stable** — Python today, possibly Rust/TS later. JCS-style
   canonicalization (sorted keys, no whitespace, UTF-8) gets us there without
   a full RFC 8785 implementation, which is overkill for the hackathon.
3. **Hex-prefixed** — matches Solidity `bytes32` literal format.

If you change this function's behavior, you invalidate every previously-pinned
trace. Bump :data:`HASHER_VERSION` and document the migration.
"""

from __future__ import annotations

import hashlib
from typing import Any

import orjson
from pydantic import BaseModel

HASHER_VERSION = "1.0.0"


def _canonical_bytes(payload: dict[str, Any] | BaseModel) -> bytes:
    """Serialize to canonical JSON: sorted keys, no whitespace, UTF-8."""
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    # orjson with OPT_SORT_KEYS gives us JCS-lite. No spaces, sorted, UTF-8.
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)


def canonical_hash(payload: dict[str, Any] | BaseModel) -> str:
    """Return ``0x``-prefixed SHA-256 hex of the canonical JSON encoding.

    Suitable as a Solidity ``bytes32`` literal.
    """
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return f"0x{digest}"


def content_hash(payload: BaseModel, *, exclude: set[str] | None = None) -> str:
    """Hash with volatile fields excluded.

    Use when you want two semantically-identical traces produced at different
    times to collide. Default-excludes ``thesis_id``, ``timestamp``, and
    ``question_id``.
    """
    default_excludes = {"thesis_id", "timestamp", "question_id"}
    excludes = default_excludes | (exclude or set())
    data = payload.model_dump(mode="json", exclude=excludes)
    return canonical_hash(data)
