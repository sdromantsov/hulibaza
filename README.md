# hulibaza

Self-hosted knowledge-base management over the
[Model Context Protocol](https://modelcontextprotocol.io). Point it at the
document folders you already keep on disk — each collection is one directory
with a small `section.yaml` — and your LLM client can ingest them and get back
**ranked source passages with provenance**. hulibaza retrieves, it does not
generate: the model composes the answer from real, cited chunks.

## Why it's built this way

- **Grounded.** Every hit is an exact passage with `source_file` + `page` +
  `chunk`. No summarizer, no hallucination surface.
- **Your tools, your files.** Sections are plain directories plus one YAML
  file — the same layout you'd keep by hand. Optional `.hulibazaignore` /
  `.hulibazaallow` (gitignore-style globs) control what gets indexed.
- **Self-hosted.** Docs, Postgres, Qdrant and the embedder all run on your
  box. Nothing leaves it.
- **Hybrid retrieval.** Dense + sparse vectors fused (RRF), so exact
  identifiers and meaning both land; an optional reranker re-orders the final
  pool.
- **Honest index.** Every section and file is fingerprinted in Postgres. If
  embedding params changed, files are mid-ingest, or the index is stale,
  search is blocked or clearly warns — you never silently search an
  inconsistent index.

## Deploy & configuration

Prereqs: Docker + Compose. A GPU is recommended for the local embedder — or
point `embedding_url` at any OpenAI-compatible endpoint and skip it.

```bash
git clone <repo> && cd hulibaza
./scripts/fetch-models.sh     # weights + tokenizers, checksum-verified
cp config.example.yaml config.yaml
cp docker-compose.yaml.example docker-compose.yaml
docker compose up -d postgres qdrant
docker compose up -d llama-server   # local embedder + reranker (optional)
docker compose up -d hulibaza       # builds the image; MCP on :59980
```

The only thing you adjust for your own knowledge base: replace the
`./YOUR_KNOWLEDGE_BASE` placeholder in `docker-compose.yaml` with the path to
your docs directory (it mounts to `/data/docs`), and set `wiki_dir` in
`config.yaml` to match. On a first start with no sections the server logs this
for you. Each subdirectory with a `section.yaml` becomes a section:

```
wiki_dir/cuda/
  section.yaml        # description + embed_model + chunk_size
  guide.pdf
  api/reference.md
```

PDFs are parsed page-aware; text is chunked structure-aware (code blocks stay
intact). Model provenance + licenses: [docs/MODELS.md](docs/MODELS.md). Full
operating notes: [docs/RUNNING.md](docs/RUNNING.md).

Then point your MCP client at `http://localhost:59980/mcp`, call
`ingest("all")`, and `search()`.

## API (MCP tools)

| Tool | What it does |
|---|---|
| `sections()` | List collections + which are usable |
| `ingest(section)` | Index a section — background, incremental, resumable |
| `search(section, query, mode)` | Ranked chunks; `hybrid` \| `semantic` \| `keyword` |
| `section_details(section)` | Config + ingestion coverage |
| `list_files` / `get_chunks` | Navigate a section's files and chunks |
| `status()` | Ingest runs, per-file errors/skips, backend health |

## Internal state

Postgres is the authority. Each **section** row stores its embedding
fingerprint (model, chunk size, overlap); each **file** row stores content
hash, size, mtime and a per-batch checkpoint. That's what makes ingest
incremental (untouched files skip), resumable (crash mid-ingest → continue
from the last committed batch), and lets the search gates detect parameter
drift and stale indexes. Deleted files become 7-day tombstones before their
vectors are purged.

## License

[MIT](LICENSE).
