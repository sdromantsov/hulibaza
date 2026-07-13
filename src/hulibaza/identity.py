"""Content-addressed identity: file content hashes and deterministic point IDs
.

A point's ID is uuid5 over (section_name, source_file, content_hash,
chunk_index). This makes upserts idempotent (same content -> same ID ->
overwrite on retry), lets a changed file's new chunks COEXIST with the old ones
until the tombstone is purged (different content_hash -> different IDs), and
separates identical-content files at different paths/sections. It removes the
need for a delete-before-upsert.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

# Fixed namespace for all hulibaza point IDs. NEVER change it: every stored
# point ID derives from it, so a new namespace would orphan all existing
# vectors. Derived deterministically (uuid5 is pure) and version-tagged so a
# future ID-scheme change is explicit.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "hulibaza.point-id.v1")

# Read files in 1 MiB blocks when hashing so large PDFs don't load fully.
_HASH_BLOCK = 1 << 20


def content_hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def content_hash_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def point_id(
    section_name: str,
    source_file: str,
    content_hash: str,
    chunk_index: int,
) -> str:
    """Deterministic uuid5 point ID for a chunk.

    Components are NUL-joined: a NUL byte cannot appear in any of them
    (filesystem paths forbid it, the hash is hex, names/ints are safe), so the
    mapping from (section, file, hash, index) to ID is unambiguous.
    """
    name = "\x00".join(
        (section_name, source_file, content_hash, str(chunk_index))
    )
    return str(uuid.uuid5(NAMESPACE, name))
