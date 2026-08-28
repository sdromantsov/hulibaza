# Running hulibaza

## Local dev / tests

```bash
pip install -e ".[dev]"

docker compose up -d postgres qdrant     # data stores (PG:59942, Qdrant:59943)

pytest                     # full suite (uses the compose Postgres)
pytest -m "not integration"  # no infra needed (unit only)
```

Integration tests auto-skip if Postgres isn't reachable. Override its URL with
`HULIBAZA_TEST_PG`.

## Running the server

The stack has its own ports/volumes/creds (never collides with the prototype):
Postgres `59942`, Qdrant `59943`, embedder `59944`, server `59980`.

1. **Data stores** — `docker compose up -d postgres qdrant`.
2. **Config** — `cp config.example.yaml config.yaml`, edit URLs / `models` / `defaults`.
3. **Tokenizers** — put each model's `tokenizer.json` in `./tokenizers/` (paths
   must match `models[*].tokenizer_path`). Token counts are computed locally.
4. **Wiki** — put sections under `./wiki/`; each section is a subdirectory with a
   `section.yaml`. Optional per-section `.hulibazaignore` / `.hulibazaallow`
   extend the shipped defaults.
5. **Embedder** — start the GPU router (`docker compose --profile gpu up -d
   llama-server`) with models in `./llama_server/`, or point `embedding_url` at
   any OpenAI-compatible `/v1/embeddings` server.

   > **Sizing the embedder for concurrency.** llama.cpp uses a *unified* KV
   > cache: the sum of all concurrently-decoded sequences must fit in
   > `ctx-size`. With `parallel = P` slots, that means **`ctx-size >= P ×
   > max_chunk_tokens`** (+ margin for special/EOS tokens). If it's too small,
   > concurrent sections — or even one section's batched request, which fans
   > out across all P slots — overflow with HTTP 500 *"Context size has been
   > exceeded"* and the whole file is dropped. The router auto-picks `P = 4`,
   > so a model serving 1024-token chunks needs `ctx-size` ~4096, or pin
   > `parallel = 2` and `ctx-size = 3072` (what this repo's `models.ini` uses).
   > This is independent of the model's native context — a chunk still must be
   > `<= ctx-size / parallel`.
6. **Server** — `docker compose --profile server up -d hulibaza`
   (streamable-HTTP MCP on `59980`). It builds the manager and starts the
   lifecycle daemon; ingestion is never auto-started.

To run the server on the host instead of in compose, set the URLs in
`config.yaml` to `localhost` with the mapped ports and run
`CONFIG_PATH=config.yaml python -m hulibaza.server`.

## Ingest / search

Ingestion and retrieval are MCP tools (`ingest`, `search`, `sections`,
`section_details`, `list_files`, `get_chunks`, `status`) — call them from an MCP
client. `ingest("all")` runs in the background; poll `status()`.

## Purge (destructive, human-only)

```bash
docker compose exec hulibaza python -m hulibaza.purge
```

Deletes tombstones past their grace (`in_use=false AND delete_after ≤ now`) from
Qdrant + Postgres. Skips any section mid-ingest. Never automatic.
