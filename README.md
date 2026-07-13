# hulibaza

**Self-hosted, grounded-retrieval MCP server.** hulibaza indexes your document
collections and serves *ranked source passages with provenance* to an LLM over
the [Model Context Protocol](https://modelcontextprotocol.io). It **retrieves,
it does not generate** — the model composes the answer from real, cited chunks.

## Why

- **Grounded, not generative.** Every hit is an exact passage with
  `source_file` + `page_number` + `chunk_index`. No summarization step, no
  hallucination surface — just retrieval.
- **Self-hosted, no data egress.** Your documents, Postgres, Qdrant, and the
  embedder all run locally. Nothing leaves the box.
- **Hybrid retrieval.** Dense semantic vectors *and* sparse keyword vectors,
  fused with Reciprocal Rank Fusion — exact identifiers and meaning both land.
- **Correct under failure.** Per-batch write-ahead checkpointing: a file
  committed as indexed always has its vectors. Crash or reboot mid-ingest →
  resume from exactly where it stopped, no corruption.
- **Honest by construction.** Consistency gates block (or clearly warn on)
  retrieval when the index is stale, mid-ingest, or the embedding params
  changed. You never silently search an inconsistent index.

## Architecture

```
              MCP client (LLM)
                    │  streamable-HTTP  :59980
              ┌─────▼──────┐
              │  hulibaza  │   FastMCP — discovery · ingest · retrieval · lifecycle
              └──┬──────┬──┘
          search │      │ state
          ┌──────▼─┐  ┌─▼───────┐          embeddings
          │ Qdrant │  │Postgres │      ┌──────────────┐
          │ hybrid │  │  state  │      │   embedder   │  /v1/embeddings
          └────────┘  └─────────┘      └──────────────┘
```

- **hulibaza server** — [FastMCP](https://modelcontextprotocol.io) over
  streamable-HTTP. Section discovery, ingestion, retrieval, and the background
  lifecycle daemon.
- **Qdrant** — vector store. Named vectors `semantic` (dense, cosine) +
  `keywords` (sparse, IDF-weighted). Deterministic point IDs make re-ingest
  idempotent.
- **Postgres** — authoritative state: per-section + per-file tracking, WAL
  checkpoints, soft-delete tombstones.
- **Embedder** — any OpenAI-compatible `/v1/embeddings` endpoint. Ships with a
  [llama.cpp](https://github.com/ggml-org/llama.cpp) multi-model router config,
  but you can point `embedding_url` anywhere.

## Quickstart

Prereqs: Docker + Compose. A GPU is recommended for a local embedder, or point
`embedding_url` at any hosted OpenAI-compatible embeddings API.

1. **Config** — `cp config.example.yaml config.yaml`, edit the model registry.
2. **Tokenizers** — drop each model's `tokenizer.json` in `./tokenizers/`
   (token counts are computed locally — no network tokenization).
3. **Embedder** — put GGUF models + a `models.ini` in `./llama_server/`, or set
   `embedding_url` to a remote endpoint and skip the `gpu` profile below.
4. **Sections** — put document collections under `./wiki/<name>/`, each with a
   `section.yaml`.
5. **Run:**
   ```bash
   docker compose up -d postgres qdrant
   docker compose --profile gpu up -d llama-server     # local embedder (optional)
   docker compose --profile server up -d hulibaza      # MCP server on :59980
   ```
6. Point your MCP client at `http://localhost:59980/mcp`, `ingest()` a section,
   then `search()` it.

Full deploy notes: [docs/RUNNING.md](docs/RUNNING.md).

## A section

```
wiki/cuda/
  section.yaml          # description + embed_model + chunk_size
  guide.pdf
  api/reference.md
```
```yaml
# section.yaml
description: NVIDIA CUDA documentation
embed_model: qwen3-embed-4b
chunk_size: 1024
```
Optional per-section `.hulibazaignore` / `.hulibazaallow` (gitignore-style
globs) extend the shipped defaults for what gets indexed. PDFs are parsed
page-aware; text files are chunked structure-aware (code blocks kept intact).

## MCP tools

| Tool | Purpose |
|---|---|
| `sections()` | List collections + which are usable. |
| `search(section, query, mode, ...)` | Ranked chunks. `mode`: `hybrid` \| `semantic` \| `keyword`. |
| `section_details(section)` | Config + ingestion coverage. |
| `list_files` / `get_chunks` | Navigate a section's files and chunks. |
| `ingest(section)` | Index a section — background, incremental, resumable. |
| `status(filters)` | Run state, per-file errors/skips, model + backend health. |

## How retrieval stays honest

Two gates protect every search:

- **Validity** — if a section's embedding parameters changed since it was
  indexed, dense modes are blocked; `keyword` still works (it needs no
  embedder).
- **Completeness** — if any file is pending, changed, or mid-ingest, all modes
  are blocked unless you pass `allow_incomplete=true`, which returns the indexed
  subset plus a warning.

## Development

```bash
pip install -e .
docker compose up -d postgres      # integration tests use a real Postgres
pytest                             # Qdrant runs in-memory; the embedder is faked
```

Design docs — requirements (FR/NFR), the decision record, and schema diagrams —
live in [docs/](docs/).

## License

[MIT](LICENSE).
