"""Tests for structure-aware text chunking (ported from the prototype to lock
in parity — this is the battle-tested fragile logic)."""

import pytest

from hulibaza.chunker import (
    chunk_document,
    chunk_pages,
    estimate_tokens,
    Chunk,
    _split_code_block,
    _split_table,
    _word_split_to_fit,
    _char_split_to_fit,
)


@pytest.mark.unit
class TestEstimateTokens:
    def test_english_short(self):
        # "hello world foo" = 3 words, 15 chars → max(3, 15//3=5) = 5
        assert estimate_tokens("hello world foo") == 5

    def test_english_long_words(self):
        # Word count dominates when words are short
        text = "a b c d e f g h i j"
        assert estimate_tokens(text) == 10  # 10 words > 19//3=6

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_russian(self):
        # Russian words are longer, char//3 should dominate
        text = "Гражданский кодекс Российской Федерации"
        tokens = estimate_tokens(text)
        assert tokens >= 4  # at least word count


@pytest.mark.unit
class TestChunkDocument:
    def test_empty_text(self):
        chunks = chunk_document("", "test.md")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_document("   \n\n   ", "test.md")
        assert chunks == []

    def test_single_short_text(self):
        chunks = chunk_document("Hello world.", "test.md", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].source_file == "test.md"
        assert chunks[0].chunk_index == 0

    def test_respects_chunk_size_in_tokens(self):
        text = " ".join(f"w{i}" for i in range(200))
        chunks = chunk_document(text, "test.md", chunk_size=50, chunk_overlap=0)
        assert len(chunks) > 1
        assert all(len(c.text.split()) <= 55 for c in chunks)

    def test_custom_token_counter(self):
        text = "Hello world. This is a longer text for testing."
        chunks = chunk_document(text, "test.md", chunk_size=10, chunk_overlap=0,
                                token_counter=lambda t: len(t))
        assert len(chunks) > 1

    def test_code_block_fits_in_chunk(self, sample_markdown):
        chunks = chunk_document(sample_markdown, "test.md", chunk_size=500, chunk_overlap=0)
        code_chunks = [c for c in chunks if "def cuda_malloc" in c.text]
        assert len(code_chunks) == 1
        assert "```python" in code_chunks[0].text
        assert "return ptr" in code_chunks[0].text

    def test_table_fits_in_chunk(self, sample_markdown):
        chunks = chunk_document(sample_markdown, "test.md", chunk_size=500, chunk_overlap=0)
        table_chunks = [c for c in chunks if "cudaMalloc" in c.text and "|" in c.text]
        assert len(table_chunks) == 1
        assert "cudaFree" in table_chunks[0].text

    def test_headings_start_new_chunk(self):
        text = "# First\n\nContent one.\n\n# Second\n\nContent two.\n"
        chunks = chunk_document(text, "test.md", chunk_size=1000, chunk_overlap=0)
        assert len(chunks) >= 2
        assert "# First" in chunks[0].text
        assert "# Second" in chunks[1].text

    def test_page_number_preserved(self):
        chunks = chunk_document("Some text.", "test.md", page_number=5)
        assert chunks[0].page_number == 5

    def test_chunk_index_sequential(self, sample_markdown):
        chunks = chunk_document(sample_markdown, "test.md", chunk_size=50, chunk_overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_token_count_property(self):
        chunks = chunk_document("one two three four five", "test.md")
        # 5 words, 23 chars → max(5, 23//3=7) = 7
        assert chunks[0].token_count == 7
        assert chunks[0].char_count == 23

    def test_deterministic_same_input(self, sample_markdown):
        # same text + config → identical chunks in identical order.
        a = chunk_document(sample_markdown, "test.md", chunk_size=60, chunk_overlap=10)
        b = chunk_document(sample_markdown, "test.md", chunk_size=60, chunk_overlap=10)
        assert [(c.text, c.page_number) for c in a] == [(c.text, c.page_number) for c in b]


@pytest.mark.unit
class TestChunkPages:
    def test_multi_page(self):
        pages = [
            ("Page one content.", 1),
            ("Page two content.", 2),
            ("Page three content.", 3),
        ]
        chunks = chunk_pages(pages, "test.pdf", chunk_size=512)
        assert len(chunks) == 3
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2
        assert chunks[2].page_number == 3

    def test_sequential_indexing(self):
        pages = [("First page.", 1), ("Second page.", 2)]
        chunks = chunk_pages(pages, "test.pdf")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_page_skipped(self):
        pages = [("Content.", 1), ("   ", 2), ("More content.", 3)]
        chunks = chunk_pages(pages, "test.pdf")
        page_numbers = [c.page_number for c in chunks]
        assert 2 not in page_numbers


@pytest.mark.unit
class TestSplitCodeBlock:
    def test_small_code_block_intact(self):
        code = "```python\nprint('hello')\n```"
        pieces = _split_code_block(code, chunk_size=100, token_counter=estimate_tokens)
        assert len(pieces) == 1
        assert pieces[0] == code

    def test_large_code_block_splits_at_lines(self):
        lines = [f"line_{i} = {i}" for i in range(100)]
        code = "```python\n" + "\n".join(lines) + "\n```"
        pieces = _split_code_block(code, chunk_size=20, token_counter=estimate_tokens)
        assert len(pieces) > 1
        for piece in pieces:
            assert piece.startswith("```python")
            assert piece.endswith("```")

    def test_no_fence(self):
        text = "just plain text\nwith lines\nand more"
        pieces = _split_code_block(text, chunk_size=3, token_counter=estimate_tokens)
        assert len(pieces) > 1


@pytest.mark.unit
class TestSplitTable:
    def test_small_table_intact(self):
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        pieces = _split_table(table, chunk_size=100, token_counter=estimate_tokens)
        assert len(pieces) == 1

    def test_large_table_splits_with_headers(self):
        header = "| Name | Value |"
        sep = "| --- | --- |"
        rows = [f"| item{i} | {i} |" for i in range(50)]
        table = "\n".join([header, sep] + rows)
        pieces = _split_table(table, chunk_size=20, token_counter=estimate_tokens)
        assert len(pieces) > 1
        for piece in pieces:
            assert piece.startswith("| Name | Value |")
            assert "| --- | --- |" in piece


@pytest.mark.unit
class TestOverlap:
    def test_overlap_present_after_heading(self):
        text = (
            "First sentence of context. Second sentence of context.\n\n"
            "# New Section\n\n"
            "Content after heading."
        )
        chunks = chunk_document(text, "test.md", chunk_size=20, chunk_overlap=5)
        if len(chunks) > 1:
            assert len(chunks[1].text) > len("# New Section\n\nContent after heading.")

    def test_zero_overlap(self):
        text = "Word " * 100
        chunks_no_overlap = chunk_document(text.strip(), "test.md", chunk_size=20, chunk_overlap=0)
        chunks_with_overlap = chunk_document(text.strip(), "test.md", chunk_size=20, chunk_overlap=5)
        total_no = sum(c.char_count for c in chunks_no_overlap)
        total_with = sum(c.char_count for c in chunks_with_overlap)
        assert total_with >= total_no


@pytest.mark.unit
class TestCharSplitFallback:
    """Verify the chunker honors chunk_size even for content with no whitespace
    breaks — long hex dumps, base64 blobs, single-line minified data. This is
    the failure the char-boundary fallback (prototype commit ca9a273) fixes."""

    def test_unbroken_token_split_at_char_boundary(self):
        blob = "0xff02ffff01ff02ffff03" * 450
        assert " " not in blob
        pieces = _char_split_to_fit(blob, chunk_size=200, token_counter=len)
        assert all(len(p) <= 200 for p in pieces)
        assert "".join(pieces) == blob
        assert len(pieces) >= len(blob) // 200

    def test_word_split_falls_back_to_chars_on_oversized_word(self):
        text = "tiny words " + ("A" * 5000) + " more tiny words"
        pieces = _word_split_to_fit(text, chunk_size=100, token_counter=len)
        assert all(len(p) <= 100 for p in pieces), \
            f"oversized piece: max={max(len(p) for p in pieces)}"

    def test_chunk_document_honors_chunk_size_on_hex_blob(self):
        blob = "0xff02ffff01ff02ffff03" * 500  # ~10500 chars, one "word"
        chunks = chunk_document(blob, "puzzle.md", chunk_size=200,
                                chunk_overlap=0, token_counter=len)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 200, f"chunk_size violated: {len(c.text)} > 200"

    def test_char_split_progresses_under_pathological_tokenizer(self):
        always_over = lambda s: 999 if s else 0
        pieces = _char_split_to_fit("abcde", chunk_size=10, token_counter=always_over)
        assert "".join(pieces) == "abcde"
        assert all(len(p) >= 1 for p in pieces)

    def test_char_split_empty_input(self):
        assert _char_split_to_fit("", chunk_size=100, token_counter=len) == []

    def test_char_split_fits_in_one_piece(self):
        pieces = _char_split_to_fit("short", chunk_size=100, token_counter=len)
        assert pieces == ["short"]
