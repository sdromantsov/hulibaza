# Models & tokenizers

hulibaza needs two things at runtime that are **not** in this repository:

- **GGUF embedding weights** — served by the llama.cpp router (`llama_server/`).
- **`tokenizer.json` files** — used for *local* token counting (`tokenizers/`),
  so chunk sizes match what the embedder actually sees.

They aren't committed because the weights (325 MB + 2.5 GB) exceed GitHub's
100 MB per-file limit, and every file is Apache-2.0 and freely downloadable from
its official HuggingFace repo. `llama_server/` and `tokenizers/` are gitignored.

## Get them

```bash
./scripts/fetch-models.sh
```

Idempotent, resumable, and it verifies each file's sha256. No HF account needed
(all repos are public). Then `cp config.example.yaml config.yaml` and bring the
stack up (see [RUNNING.md](RUNNING.md)).

## Provenance

Everything below is **Apache-2.0**.

### Weights → `llama_server/models/`

| Local file | Source repo | Quant | Size |
|---|---|---|---|
| `nomic-embed-text-v2-moe.Q4_K_S.gguf` | [nomic-ai/nomic-embed-text-v2-moe-GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF) | Q4_K_S | 325 MB |
| `Qwen3-Embedding-4B-Q4_K_M.gguf` | [Qwen/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) | Q4_K_M | 2.5 GB |

### Tokenizers → `tokenizers/`

Each is the model repo's `tokenizer.json`, renamed to the id `config.yaml` uses.

| Local file | Source repo → file |
|---|---|
| `nomic-v2-moe.json` | [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) → `tokenizer.json` |
| `qwen3-embed-4b.json` | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) → `tokenizer.json` |

> The tokenizer must come from the **base model** repo (not the GGUF repo) and
> match the served weights — the whole point is that local token counts equal
> the embedder's. Underlying models: nomic `nomic-bert-moe`, 8×277M MoE, native
> ctx 512, dim 768 · Qwen3 4B, ctx 40960, dim 2560.

## Integrity (sha256)

```
db0608a87a2daf4a52b74912dd678ca6122db26d971bdfcf16d3b11b77047663  nomic-embed-text-v2-moe.Q4_K_S.gguf
2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b  Qwen3-Embedding-4B-Q4_K_M.gguf
3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20  nomic-v2-moe.json
83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d  qwen3-embed-4b.json
```

## Notes

- **Reproducibility:** the fetch script pulls from `main`. To pin exact bytes,
  replace `main` with a commit revision in the `resolve/<rev>/` URL.
- **Swapping models:** any OpenAI-compatible `/v1/embeddings` endpoint works —
  point `embedding_url` elsewhere and adjust the `models` registry + tokenizers
  in `config.yaml`. You are not tied to these two.
- **`nomic-v1.json`** (nomic-embed-text-v1 WordPiece tokenizer) is a leftover
  from earlier experiments and is not referenced by any config — safe to ignore.
