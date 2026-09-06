#!/usr/bin/env bash
#
# Fetch the embedding models + tokenizers hulibaza needs.
#
# These files are intentionally NOT committed: the GGUF weights are far larger
# than GitHub's 100 MB limit, and every file here is freely available from its
# official HuggingFace repo. This script pulls each one into
# the exact path config.yaml / llama_server/models.ini expect, and verifies its
# sha256. Re-runnable: existing, correct files are skipped.
#
# Requires: bash, curl, sha256sum. (No HF account needed — all repos are public.)
# Alternative: `huggingface-cli download <repo> <file>` does the same with resume.
#
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

MODELS=llama_server/models
TOKS=tokenizers
mkdir -p "$MODELS" "$TOKS"

# dest | huggingface repo | path-in-repo | sha256
ENTRIES=(
  "$MODELS/embeddinggemma-300M-Q8_0.gguf|unsloth/embeddinggemma-300m-GGUF|embeddinggemma-300M-Q8_0.gguf|a0f7b4e13c397a6e1b32c2de75b1f65a14c92ec524d5f674d94a4290a1c4969b"
  "$MODELS/qwen3-reranker-0.6b-q8_0.gguf|ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF|qwen3-reranker-0.6b-q8_0.gguf|22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"
  "$TOKS/embeddinggemma-300m.json|unsloth/embeddinggemma-300m|tokenizer.json|6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e"
)

verify() { echo "$2  $1" | sha256sum -c --status; }

for e in "${ENTRIES[@]}"; do
  IFS='|' read -r dest repo path sum <<<"$e"
  if [ -f "$dest" ] && verify "$dest" "$sum"; then
    echo "  ✓ $dest (present, checksum OK)"
    continue
  fi
  echo "  ↓ $repo :: $path"
  curl -fL -C - --retry 3 --retry-delay 2 --progress-bar \
    "https://huggingface.co/$repo/resolve/main/$path" -o "$dest.part"
  mv "$dest.part" "$dest"
  if verify "$dest" "$sum"; then
    echo "  ✓ $dest (checksum OK)"
  else
    echo "  ✗ CHECKSUM MISMATCH for $dest — the upstream file may have changed." >&2
    echo "    Compare against docs/MODELS.md and re-check the source repo." >&2
    exit 1
  fi
done

echo "All models + tokenizers in place. See docs/MODELS.md for provenance."
