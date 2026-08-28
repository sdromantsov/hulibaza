#!/usr/bin/env bash
#
# Fetch the embedding models + tokenizers hulibaza needs.
#
# These files are intentionally NOT committed: the GGUF weights are far larger
# than GitHub's 100 MB limit, and every file here is Apache-2.0 and freely
# available from its official HuggingFace repo. This script pulls each one into
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
  "$MODELS/nomic-embed-text-v2-moe.Q4_K_S.gguf|nomic-ai/nomic-embed-text-v2-moe-GGUF|nomic-embed-text-v2-moe.Q4_K_S.gguf|db0608a87a2daf4a52b74912dd678ca6122db26d971bdfcf16d3b11b77047663"
  "$MODELS/Qwen3-Embedding-4B-Q4_K_M.gguf|Qwen/Qwen3-Embedding-4B-GGUF|Qwen3-Embedding-4B-Q4_K_M.gguf|2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b"
  "$TOKS/nomic-v2-moe.json|nomic-ai/nomic-embed-text-v2-moe|tokenizer.json|3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20"
  "$TOKS/qwen3-embed-4b.json|Qwen/Qwen3-Embedding-4B|tokenizer.json|83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d"
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
