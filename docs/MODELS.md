# Models & tokenizers

hulibaza needs two things at runtime that are **not** in this repository:

- **GGUF weights** — embedding + reranker models, served by the llama.cpp
  router (`llama_server/`).
- **`tokenizer.json` files** — used for *local* token counting (`tokenizers/`),
  so chunk sizes match what the embedder actually sees.

They aren't committed because the weights (333 MB + 639 MB) exceed
GitHub's 100 MB per-file limit, and every file is freely downloadable from its
official HuggingFace repo (licenses per model, below). `llama_server/` and
`tokenizers/` are gitignored.

## Get them

```bash
./scripts/fetch-models.sh
```

Idempotent, resumable, and it verifies each file's sha256. No HF account needed
(all repos are public). Then `cp config.example.yaml config.yaml` and bring the
stack up (see [RUNNING.md](RUNNING.md)).

## Provenance

### Weights → `llama_server/models/`

| Local file | Source repo | Quant | Size | License |
|---|---|---|---|---|
| `embeddinggemma-300M-Q8_0.gguf` | [unsloth/embeddinggemma-300m-GGUF](https://huggingface.co/unsloth/embeddinggemma-300m-GGUF) | Q8_0 | 333 MB | [Gemma terms](https://ai.google.dev/gemma/terms) |
| `qwen3-reranker-0.6b-q8_0.gguf` | [ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | Q8_0 | 639 MB | Apache-2.0 |

The reranker is a **species-2** model (causal LM judging yes/no relevance); the
ggml-org conversion bakes the 2-row classifier head + `rerank` template into the
GGUF, so llama.cpp's `--reranking` (router ini key `reranking = true`) serves it
at `/v1/rerank` with P(yes) scores. Requires llama.cpp ≥ v0.4.0-era builds
(`server-cuda12-b10818` pinned in `docker-compose.yaml`).

### Tokenizers → `tokenizers/`

Each is the model repo's `tokenizer.json`, renamed to the id `config.yaml` uses.
Rerankers need no local tokenizer (documents are already chunked; rerank batches
are per-document, not per-token).

| Local file | Source repo → file |
|---|---|
| `embeddinggemma-300m.json` | [unsloth/embeddinggemma-300m](https://huggingface.co/unsloth/embeddinggemma-300m) → `tokenizer.json` (mirror of the [gated](https://huggingface.co/google/embeddinggemma-300m) google repo) |

> The tokenizer must come from the **base model** repo (not the GGUF repo) and
> match the served weights — the whole point is that local token counts equal
> the embedder's. EmbeddingGemma: `gemma-3-270m` family, ctx 2048, dim 768.

## Integrity (sha256)

```
a0f7b4e13c397a6e1b32c2de75b1f65a14c92ec524d5f674d94a4290a1c4969b  embeddinggemma-300M-Q8_0.gguf
22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48  qwen3-reranker-0.6b-q8_0.gguf
6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e  embeddinggemma-300m.json
```

## Notes

- **Reproducibility:** the fetch script pulls from `main`. To pin exact bytes,
  replace `main` with a commit revision in the `resolve/<rev>/` URL.
- **Swapping models:** any OpenAI-compatible `/v1/embeddings` endpoint works —
  point `embedding_url` elsewhere and adjust the `models` registry + tokenizers
  in `config.yaml`. Rerankers need any `/v1/rerank` implementation (Jina/TEI
  format: `query` + `documents` → `relevance_score`); the same URL serves both
  when it's the llama.cpp router. You are not tied to these models.
- **`qwen3-embed-4b.json`** is a leftover from an earlier experiment (after
  Qwen3-Embedding-4B left the stack); it is not referenced by any config —
  safe to ignore or delete.
