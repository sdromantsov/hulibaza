"""Async Qdrant vector store: hybrid (dense + sparse) retrieval.

One section = one collection = one model. Each point has two named vectors —
`semantic` (dense, cosine) and `keywords` (sparse, IDF applied server-side) —
and a payload of text + provenance + the `in_use` lifecycle flag. Search always
filters `in_use = true` so tombstoned (soft-deleted / superseded) points never
surface.

Point IDs are the deterministic uuid5 from identity.point_id, so re-upserting an
unchanged chunk overwrites in place (idempotent), and a changed file's new
chunks coexist with the old until purge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    Range,
    SparseVector as QdrantSparseVector,
    SparseVectorParams,
    VectorParams,
)

from hulibaza.sparse import SparseVector

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "semantic"
SPARSE_VECTOR_NAME = "keywords"
_UPSERT_BATCH = 100
_SCROLL_BATCH = 1000


@dataclass
class ChunkPoint:
    """Everything needed to write one chunk as a Qdrant point."""

    id: str  # deterministic uuid5 (identity.point_id)
    text: str
    dense: list[float]
    sparse: SparseVector
    source_file: str
    page_number: int
    chunk_index: int
    section_name: str
    in_use: bool = True


@dataclass
class SearchResult:
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    section_name: str
    score: float


@dataclass
class FileInfo:
    source_file: str
    total_chunks: int
    max_page: int


@dataclass
class ChunkInfo:
    text: str
    source_file: str
    page_number: int
    chunk_index: int


def _in_use_filter() -> Filter:
    return Filter(must=[FieldCondition(key="in_use", match=MatchValue(value=True))])


def _to_result(point) -> SearchResult:
    p = point.payload
    return SearchResult(
        text=p["text"],
        source_file=p["source_file"],
        page_number=p["page_number"],
        chunk_index=p["chunk_index"],
        section_name=p["section_name"],
        score=point.score,
    )


class QdrantStore:
    def __init__(self, url: str | None = None, *, client: AsyncQdrantClient | None = None) -> None:
        # An injected client (e.g. AsyncQdrantClient(location=":memory:")) lets
        # tests run the real local engine with no server.
        if client is not None:
            self.client = client
        elif url is not None:
            self.client = AsyncQdrantClient(url=url)
        else:
            raise ValueError("QdrantStore needs a url or a client")

    async def aclose(self) -> None:
        await self.client.close()

    # ── collection lifecycle ──

    async def collection_exists(self, collection: str) -> bool:
        return await self.client.collection_exists(collection)

    async def ensure_collection(self, collection: str, vector_size: int) -> None:
        if await self.collection_exists(collection):
            logger.info("Collection '%s' already exists", collection)
            return
        await self.client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        logger.info("Created collection '%s' (dim=%d)", collection, vector_size)

    async def drop_collection(self, collection: str) -> None:
        if await self.collection_exists(collection):
            await self.client.delete_collection(collection)
            logger.info("Dropped collection '%s'", collection)

    # ── writes ──

    async def upsert_chunks(self, collection: str, points: list[ChunkPoint]) -> int:
        if not points:
            return 0
        structs = [
            PointStruct(
                id=cp.id,
                vector={
                    DENSE_VECTOR_NAME: cp.dense,
                    SPARSE_VECTOR_NAME: QdrantSparseVector(
                        indices=cp.sparse.indices, values=cp.sparse.values
                    ),
                },
                payload={
                    "text": cp.text,
                    "source_file": cp.source_file,
                    "page_number": cp.page_number,
                    "chunk_index": cp.chunk_index,
                    "section_name": cp.section_name,
                    "in_use": cp.in_use,
                },
            )
            for cp in points
        ]
        for i in range(0, len(structs), _UPSERT_BATCH):
            await self.client.upsert(
                collection_name=collection,
                points=structs[i : i + _UPSERT_BATCH],
                wait=True,
            )
        logger.info("Upserted %d points to '%s'", len(structs), collection)
        return len(structs)

    async def delete_by_file(self, collection: str, source_file: str) -> None:
        await self.client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
            wait=True,
        )
        logger.info("Deleted points for '%s' from '%s'", source_file, collection)

    async def set_in_use(self, collection: str, source_file: str, value: bool) -> None:
        """Flip the in_use flag for all of a file's points (tombstone / restore)."""
        await self.client.set_payload(
            collection_name=collection,
            payload={"in_use": value},
            points=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
            wait=True,
        )

    async def get_file_points(self, collection: str, source_file: str) -> list[ChunkPoint]:
        """Fetch a file's points WITH vectors — used to repath a moved file by
        rewriting its points under new (source_file-derived) IDs without
        re-embedding."""
        out: list[ChunkPoint] = []
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
                ),
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                qsparse = p.vector[SPARSE_VECTOR_NAME]
                out.append(
                    ChunkPoint(
                        id=str(p.id),
                        text=p.payload["text"],
                        dense=list(p.vector[DENSE_VECTOR_NAME]),
                        sparse=SparseVector(indices=list(qsparse.indices), values=list(qsparse.values)),
                        source_file=p.payload["source_file"],
                        page_number=p.payload["page_number"],
                        chunk_index=p.payload["chunk_index"],
                        section_name=p.payload["section_name"],
                        in_use=p.payload["in_use"],
                    )
                )
            if offset is None:
                break
        return out

    # ── search (always in_use=true) ──

    async def hybrid_search(
        self, collection: str, dense: list[float], sparse: SparseVector, limit: int = 3
    ) -> list[SearchResult]:
        qsparse = QdrantSparseVector(indices=sparse.indices, values=sparse.values)
        flt = _in_use_filter()
        results = await self.client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense, using=DENSE_VECTOR_NAME, limit=limit * 5, filter=flt),
                Prefetch(query=qsparse, using=SPARSE_VECTOR_NAME, limit=limit * 5, filter=flt),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
        return [_to_result(p) for p in results.points]

    async def semantic_search(
        self, collection: str, dense: list[float], limit: int = 3
    ) -> list[SearchResult]:
        results = await self.client.query_points(
            collection_name=collection,
            query=dense,
            using=DENSE_VECTOR_NAME,
            query_filter=_in_use_filter(),
            limit=limit,
        )
        return [_to_result(p) for p in results.points]

    async def keyword_search(
        self, collection: str, sparse: SparseVector, limit: int = 3
    ) -> list[SearchResult]:
        qsparse = QdrantSparseVector(indices=sparse.indices, values=sparse.values)
        results = await self.client.query_points(
            collection_name=collection,
            query=qsparse,
            using=SPARSE_VECTOR_NAME,
            query_filter=_in_use_filter(),
            limit=limit,
        )
        return [_to_result(p) for p in results.points]

    # ── read helpers (status / get_chunks) ──

    async def list_files(self, collection: str) -> list[FileInfo]:
        """Aggregate in_use points by source_file (payload-only scroll)."""
        files: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=collection,
                scroll_filter=_in_use_filter(),
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=["source_file", "page_number"],
                with_vectors=False,
            )
            for p in points:
                fs = p.payload["source_file"]
                pn = p.payload["page_number"]
                entry = files.get(fs)
                if entry is None:
                    files[fs] = {"chunks": 1, "max_page": pn}
                else:
                    entry["chunks"] += 1
                    entry["max_page"] = max(entry["max_page"], pn)
            if offset is None:
                break
        return [
            FileInfo(source_file=fs, total_chunks=v["chunks"], max_page=v["max_page"])
            for fs, v in sorted(files.items())
        ]

    async def get_chunks_by_index(
        self, collection: str, source_file: str, start: int, count: int
    ) -> list[ChunkInfo]:
        if count <= 0:
            return []
        points, _ = await self.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                    FieldCondition(key="chunk_index", range=Range(gte=start, lt=start + count)),
                    FieldCondition(key="in_use", match=MatchValue(value=True)),
                ]
            ),
            limit=count,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [
            ChunkInfo(
                text=p.payload["text"],
                source_file=p.payload["source_file"],
                page_number=p.payload["page_number"],
                chunk_index=p.payload["chunk_index"],
            )
            for p in points
        ]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    async def count(self, collection: str) -> int:
        try:
            return (await self.client.count(collection, exact=True)).count
        except Exception:
            return 0

    async def health_check(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False
