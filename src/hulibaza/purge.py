"""Destructive, human-only purge.

A standalone DB-only process: for each section it can lock, it deletes every
tombstone whose grace has elapsed (`in_use=false AND delete_after <= now`) from
Qdrant, then removes the PG row. It acquires the section advisory lock and skips
any section mid-ingest (or being purged elsewhere). Never automatic, never
LLM-triggered.

Run:  python -m hulibaza.purge
"""

from __future__ import annotations

import asyncio
import logging
import sys

from hulibaza.config import load_global_config
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

logger = logging.getLogger(__name__)


async def run_purge(state: StateStore, qdrant: QdrantStore) -> dict:
    """Purge every eligible tombstone across all sections. Returns per-section
    outcome ({section: [purged rel_paths] | 'skipped (locked)'})."""
    expired = await state.list_expired_tombstones()  # all sections
    sections = sorted({r.section_name for r in expired})
    results: dict[str, object] = {}
    for section_name in sections:
        async with state.try_section_lock(section_name) as acquired:
            if not acquired:
                results[section_name] = "skipped (locked)"
                continue
            # Re-fetch under the lock so we act on a stable view.
            rows = await state.list_expired_tombstones(section_name)
            purged = []
            for r in rows:
                await qdrant.delete_by_file(section_name, r.rel_path)
                await state.delete_file_row(section_name, r.rel_path)
                purged.append(r.rel_path)
            results[section_name] = purged
            logger.info("Purged %d tombstone(s) from '%s'", len(purged), section_name)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_global_config()
    state = StateStore(config.postgres_url)
    qdrant = QdrantStore(config.qdrant_url)
    results = asyncio.run(run_purge(state, qdrant))
    total = sum(len(v) for v in results.values() if isinstance(v, list))
    print(f"Purged {total} tombstone(s) across {len(results)} section(s):")
    for section, outcome in results.items():
        if isinstance(outcome, list):
            print(f"  {section}: {len(outcome)} removed")
        else:
            print(f"  {section}: {outcome}")


if __name__ == "__main__":
    main()
