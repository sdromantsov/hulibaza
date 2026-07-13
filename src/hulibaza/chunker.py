"""Structure-aware text chunking for technical documents.

All size measurements (chunk_size, chunk_overlap) are in tokens. The token
counter is injectable; production passes the section model's local tokenizer.
The default `estimate_tokens` is a conservative multilingual heuristic
(max(word_count, char_count // 3)) used only when no counter is supplied.

Chunking is deterministic: the same text + config + counter always produce the
same chunks in the same order. The resume + content-addressed-ID design
 relies on this.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Estimate token count conservatively.

    Uses max(word_count, char_count // 3) to handle both Latin and
    Cyrillic/multilingual text. For English, word count dominates
    (~1 word ≈ 1-1.5 tokens). For Russian and other languages where
    words tokenize to multiple subwords, char_count // 3 dominates
    (~3 chars ≈ 1 token is conservative). Always overestimates real
    token count (safe direction).
    """
    if not text:
        return 0
    return max(len(text.split()), len(text) // 3)


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_file: str
    page_number: int

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.text)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class _Block:
    text: str
    block_type: str  # "code", "table", "heading", "paragraph"
    page_number: int


_FENCED_CODE_RE = re.compile(
    r"^(`{3,}|~{3,})[^\n]*\n(.*?)\n\1\s*$",
    re.MULTILINE | re.DOTALL,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_document(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    page_number: int = 0,
    token_counter: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    if not text.strip():
        return []

    blocks = _split_into_blocks(text, page_number)
    chunks = _merge_blocks_into_chunks(blocks, source_file, chunk_size, chunk_overlap, token_counter)
    chunks = _validate_and_resplit(chunks, chunk_size, token_counter)

    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    return chunks


def _validate_and_resplit(
    chunks: list[Chunk],
    chunk_size: int,
    token_counter: TokenCounter,
) -> list[Chunk]:
    """Verify every final chunk's real token count and split any that overshoot.

    The in-progress chunker counts tokens per block or per sentence and sums
    them, which can drift ~5-10% from the real token count of the joined
    chunk (due to separator tokens, SentencePiece boundary effects, or
    server-side BOS/EOS). This pass catches the drift by tokenizing each
    assembled chunk exactly and, if it exceeds chunk_size, word-splits it
    to pieces that each fit.
    """
    validated: list[Chunk] = []
    for chunk in chunks:
        real = token_counter(chunk.text)
        if real <= chunk_size:
            validated.append(chunk)
            continue
        logger.debug(
            "Chunk overshoot for %s: %d tokens > chunk_size=%d, resplitting",
            chunk.source_file, real, chunk_size,
        )
        for piece in _word_split_to_fit(chunk.text, chunk_size, token_counter):
            validated.append(Chunk(
                text=piece, chunk_index=0,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
            ))
    return validated


def _word_split_to_fit(
    text: str,
    chunk_size: int,
    token_counter: TokenCounter,
) -> list[str]:
    """Force-split text into pieces each ≤ chunk_size tokens.

    Tries word boundaries first. When a single "word" (whitespace-bounded
    run) is itself larger than chunk_size — e.g. an unbroken hex bytecode
    dump, a base64 blob, a long URL — falls back to character-boundary
    splitting for that word. This is the chunker's last line of defense:
    the post-condition is that no emitted chunk exceeds chunk_size,
    regardless of source content.
    """
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []

    def _flush_current() -> None:
        if current:
            pieces.append(" ".join(current))
            current.clear()

    for word in words:
        # A single oversized token has no word boundaries to exploit —
        # break it at character boundaries instead.
        if token_counter(word) > chunk_size:
            _flush_current()
            pieces.extend(_char_split_to_fit(word, chunk_size, token_counter))
            continue
        candidate = " ".join(current + [word])
        if token_counter(candidate) > chunk_size and current:
            _flush_current()
            current.append(word)
        else:
            current.append(word)
    _flush_current()
    return pieces


def _char_split_to_fit(
    text: str,
    chunk_size: int,
    token_counter: TokenCounter,
) -> list[str]:
    """Split ``text`` at character boundaries so each piece tokenizes to
    ``<= chunk_size`` tokens.

    Last-resort fallback for content with no whitespace breaks (hex dumps,
    base64 blobs, single-line minified data). Uses binary search to find
    the longest character prefix that fits — O(log N) tokenizer calls per
    piece, which is acceptable for the rare oversized-word case.

    Always makes at least one character of progress per iteration, so even
    a pathological tokenizer where a single char counts as ``> chunk_size``
    tokens still terminates (each char becomes its own piece).
    """
    if not text:
        return []
    pieces: list[str] = []
    remaining = text
    while remaining:
        # If the whole remainder already fits, we're done.
        if token_counter(remaining) <= chunk_size:
            pieces.append(remaining)
            break
        # Binary-search the longest prefix that fits within chunk_size.
        lo, hi = 1, len(remaining)
        best = 1  # guarantee forward progress
        while lo <= hi:
            mid = (lo + hi) // 2
            if token_counter(remaining[:mid]) <= chunk_size:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        pieces.append(remaining[:best])
        remaining = remaining[best:]
    return pieces


def chunk_pages(
    pages: list[tuple[str, int]],
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    token_counter: TokenCounter = estimate_tokens,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for text, page_num in pages:
        page_chunks = chunk_document(
            text=text,
            source_file=source_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page_num,
            token_counter=token_counter,
        )
        all_chunks.extend(page_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    return all_chunks


def _split_into_blocks(text: str, page_number: int) -> list[_Block]:
    blocks: list[_Block] = []

    # Replace fenced code blocks with placeholders
    code_blocks: dict[str, str] = {}
    code_counter = 0

    def _replace_fenced(match: re.Match) -> str:
        nonlocal code_counter
        key = f"\x00CODE_{code_counter}\x00"
        code_blocks[key] = match.group(0)
        code_counter += 1
        return key

    processed = _FENCED_CODE_RE.sub(_replace_fenced, text)

    lines = processed.split("\n")
    current_paragraph: list[str] = []
    table_lines: list[str] = []

    def _flush_paragraph():
        nonlocal current_paragraph
        if not current_paragraph:
            return
        para_text = "\n".join(current_paragraph).strip()
        if not para_text:
            current_paragraph = []
            return

        # Check for code block placeholders embedded in paragraph
        remaining = para_text
        for key, code_text in list(code_blocks.items()):
            if key in remaining:
                before, after = remaining.split(key, 1)
                if before.strip():
                    blocks.append(_Block(text=before.strip(), block_type="paragraph", page_number=page_number))
                blocks.append(_Block(text=code_text, block_type="code", page_number=page_number))
                remaining = after.strip()

        if remaining:
            blocks.append(_Block(text=remaining, block_type="paragraph", page_number=page_number))
        current_paragraph = []

    def _flush_table():
        nonlocal table_lines
        if table_lines:
            table_text = "\n".join(table_lines)
            blocks.append(_Block(text=table_text, block_type="table", page_number=page_number))
            table_lines = []

    in_table = False
    for line in lines:
        stripped = line.strip()

        # Code placeholder on its own line
        if stripped in code_blocks:
            _flush_paragraph()
            _flush_table()
            in_table = False
            blocks.append(_Block(text=code_blocks[stripped], block_type="code", page_number=page_number))
            continue

        # Heading
        if _HEADING_RE.match(stripped):
            _flush_paragraph()
            _flush_table()
            in_table = False
            blocks.append(_Block(text=stripped, block_type="heading", page_number=page_number))
            continue

        # Table row
        if _TABLE_ROW_RE.match(stripped):
            _flush_paragraph()
            table_lines.append(stripped)
            in_table = True
            continue

        # End of table
        if in_table:
            _flush_table()
            in_table = False

        current_paragraph.append(line)

    _flush_paragraph()
    _flush_table()

    return blocks


def _merge_blocks_into_chunks(
    blocks: list[_Block],
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: TokenCounter,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_size: int = 0
    current_page: int = 0

    def _flush_current():
        nonlocal current_texts, current_size
        if current_texts:
            merged = "\n\n".join(current_texts).strip()
            if merged:
                chunks.append(Chunk(
                    text=merged,
                    chunk_index=0,
                    source_file=source_file,
                    page_number=current_page,
                ))
            current_texts = []
            current_size = 0

    def _compute_overlap_prefix() -> str:
        if not chunks or chunk_overlap <= 0:
            return ""
        last_text = chunks[-1].text
        last_tokens = token_counter(last_text)
        if last_tokens <= chunk_overlap:
            return last_text
        # Take last chunk_overlap+30 words as buffer for sentence splitting
        words = last_text.split()
        tail = " ".join(words[-(chunk_overlap + 30):])
        sentences = _SENTENCE_END_RE.split(tail)
        if len(sentences) <= 1:
            # No sentence boundaries — take last chunk_overlap words
            overlap_words = words[-chunk_overlap:]
            if len(overlap_words) > 1:
                return " ".join(overlap_words[1:])
            return ""
        # Tokenize each sentence once, then sum locally. Approximation is
        # close enough for overlap computation and avoids retokenizing a
        # growing concatenated string.
        sent_tokens = [token_counter(s) for s in sentences]
        collected: list[str] = []
        total = 0
        for sent, toks in zip(reversed(sentences), reversed(sent_tokens)):
            if total + toks > chunk_overlap + 15:
                break
            collected.append(sent)
            total += toks
        return " ".join(reversed(collected)).strip()

    for block in blocks:
        if block.block_type == "code":
            _flush_current()
            for piece in _split_code_block(block.text, chunk_size, token_counter):
                chunks.append(Chunk(
                    text=piece, chunk_index=0,
                    source_file=source_file, page_number=block.page_number,
                ))
            continue

        if block.block_type == "table":
            _flush_current()
            for piece in _split_table(block.text, chunk_size, token_counter):
                chunks.append(Chunk(
                    text=piece, chunk_index=0,
                    source_file=source_file, page_number=block.page_number,
                ))
            continue

        if block.block_type == "heading":
            _flush_current()
            overlap = _compute_overlap_prefix()
            if overlap:
                current_texts.append(overlap)
                current_size = token_counter(overlap)
            current_texts.append(block.text)
            current_size += token_counter(block.text)
            current_page = block.page_number
            continue

        # Paragraph
        current_page = block.page_number
        block_text = block.text
        block_tokens = token_counter(block_text)

        # Split long paragraph blocks that exceed chunk_size on their own
        if block_tokens > chunk_size:
            _flush_current()
            para_chunks = _split_long_text(block_text, chunk_size, chunk_overlap, token_counter)
            for pc in para_chunks:
                chunks.append(Chunk(
                    text=pc, chunk_index=0,
                    source_file=source_file, page_number=block.page_number,
                ))
            continue

        if current_size + block_tokens > chunk_size and current_texts:
            _flush_current()
            overlap = _compute_overlap_prefix()
            if overlap:
                current_texts.append(overlap)
                current_size = token_counter(overlap)

        current_texts.append(block_text)
        current_size += block_tokens

    _flush_current()

    return chunks


def _split_long_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: TokenCounter,
) -> list[str]:
    """Split a long paragraph into pieces at sentence or word boundaries."""
    sentences = _SENTENCE_END_RE.split(text)
    if len(sentences) <= 1:
        # No sentence boundaries — split by words
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            if current_len + 1 > chunk_size and current:
                pieces.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += 1
        if current:
            pieces.append(" ".join(current))
        return pieces

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        sent_tokens = token_counter(sent)
        if current_len + sent_tokens > chunk_size and current:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += sent_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_code_block(
    text: str,
    chunk_size: int,
    token_counter: TokenCounter,
) -> list[str]:
    """Split a code block at line boundaries, preserving fence markers."""
    if token_counter(text) <= chunk_size:
        return [text]

    lines = text.split("\n")
    # Detect fence lines (``` or ~~~)
    open_fence = ""
    close_fence = ""
    if lines and re.match(r"^(`{3,}|~{3,})", lines[0]):
        open_fence = lines[0]
        lines = lines[1:]
    if lines and re.match(r"^(`{3,}|~{3,})\s*$", lines[-1]):
        close_fence = lines[-1]
        lines = lines[:-1]

    fence_tokens = token_counter(open_fence) + token_counter(close_fence) if open_fence else 0
    available = max(chunk_size - fence_tokens, 1)

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_tokens = token_counter(line)
        if current_len + line_tokens > available and current:
            body = "\n".join(current)
            if open_fence:
                body = open_fence + "\n" + body + "\n" + close_fence
            pieces.append(body)
            current = []
            current_len = 0
        current.append(line)
        current_len += line_tokens

    if current:
        body = "\n".join(current)
        if open_fence:
            body = open_fence + "\n" + body + "\n" + close_fence
        pieces.append(body)

    return pieces


def _split_table(
    text: str,
    chunk_size: int,
    token_counter: TokenCounter,
) -> list[str]:
    """Split a table at row boundaries, repeating header + separator."""
    if token_counter(text) <= chunk_size:
        return [text]

    lines = text.split("\n")
    if len(lines) < 3:
        return [text]

    # First two lines are header + separator (e.g. |---|---|)
    header = lines[0]
    separator = lines[1] if re.match(r"^\|[\s\-:]+\|", lines[1]) else None

    if separator:
        header_lines = [header, separator]
        data_lines = lines[2:]
    else:
        header_lines = [header]
        data_lines = lines[1:]

    header_text = "\n".join(header_lines)
    header_tokens = token_counter(header_text)
    available = max(chunk_size - header_tokens, 1)

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for row in data_lines:
        row_tokens = token_counter(row)
        if current_len + row_tokens > available and current:
            pieces.append(header_text + "\n" + "\n".join(current))
            current = []
            current_len = 0
        current.append(row)
        current_len += row_tokens

    if current:
        pieces.append(header_text + "\n" + "\n".join(current))

    return pieces
