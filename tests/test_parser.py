"""Tests for text extraction. Discovery/classification is covered in
test_files.py; this file exercises parse_file / PDF page handling only."""

import pytest

from hulibaza.parser import parse_file, ParsedDocument


@pytest.mark.unit
class TestParseText:
    def test_parses_md(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nWorld\n")
        doc = parse_file(md, tmp_path)
        assert doc.source_file == "test.md"
        assert len(doc.pages) == 1
        assert doc.pages[0].page_number == 0
        assert "Hello" in doc.full_text

    def test_preserves_structure(self, tmp_path, sample_markdown):
        md = tmp_path / "structured.md"
        md.write_text(sample_markdown)
        doc = parse_file(md, tmp_path)
        assert "```python" in doc.full_text
        assert "cudaMalloc" in doc.full_text

    def test_parses_txt(self, tmp_path, sample_text):
        txt = tmp_path / "test.txt"
        txt.write_text(sample_text)
        doc = parse_file(txt, tmp_path)
        assert doc.source_file == "test.txt"
        assert "CUDA" in doc.full_text
        assert doc.pages[0].page_number == 0

    def test_unknown_extension_parsed_as_text(self, tmp_path):
        rst = tmp_path / "guide.rst"
        rst.write_text("reStructuredText content here")
        doc = parse_file(rst, tmp_path)
        assert doc.source_file == "guide.rst"
        assert "reStructuredText" in doc.full_text
        assert doc.pages[0].page_number == 0

    def test_source_file_is_posix_relative(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        f = sub / "note.md"
        f.write_text("nested content")
        doc = parse_file(f, tmp_path)
        assert doc.source_file == "a/b/note.md"

    def test_invalid_utf8_replaced_not_crashed(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes(b"caf\xe9 valid ascii tail")  # 0xe9 = é in latin-1, bad utf-8
        doc = parse_file(f, tmp_path)
        assert "valid ascii tail" in doc.full_text  # errors="replace", no crash


@pytest.mark.unit
class TestParseFile:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_file(tmp_path / "nonexistent.md", tmp_path)


def _make_pdf(path, page_texts):
    """Write a PDF with one page per entry (empty string → blank page)."""
    import pymupdf

    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(str(path))
    finally:
        doc.close()


@pytest.mark.unit
class TestParsePDF:
    def test_parses_pages_1_indexed(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf, ["First page text.", "Second page text."])
        doc = parse_file(pdf, tmp_path)
        assert doc.source_file == "doc.pdf"
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2
        assert "First page" in doc.pages[0].text
        assert "Second page" in doc.pages[1].text

    def test_empty_page_skipped_but_numbering_preserved(self, tmp_path):
        # Page 2 is blank: it must be dropped, but page 3 must keep number 3.
        pdf = tmp_path / "gap.pdf"
        _make_pdf(pdf, ["Content.", "", "More content."])
        doc = parse_file(pdf, tmp_path)
        nums = [p.page_number for p in doc.pages]
        assert nums == [1, 3]

    def test_deterministic(self, tmp_path):
        pdf = tmp_path / "d.pdf"
        _make_pdf(pdf, ["Repeatable extraction text."])
        a = parse_file(pdf, tmp_path)
        b = parse_file(pdf, tmp_path)
        assert [p.text for p in a.pages] == [p.text for p in b.pages]
