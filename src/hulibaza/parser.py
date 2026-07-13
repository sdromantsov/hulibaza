"""Text extraction from documents.

PDFs are parsed page-by-page via PyMuPDF (page boundaries preserved). Every
other file is read as UTF-8 text as a single page (page_number = 0). WHICH files
reach here is decided by files.py; this module only extracts.

Extraction is deterministic — the same bytes always yield the same pages in the
same order — which the resume + content-addressed-ID design
depends on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions handled by a dedicated binary parser; everything else is read as
# UTF-8 text. Single source of truth for the "known-binary" class — files.py
# imports this to classify files.
KNOWN_BINARY_EXTS: frozenset[str] = frozenset({".pdf"})


@dataclass
class ParsedPage:
    text: str
    page_number: int  # 1-indexed for PDF; 0 for non-paginated


@dataclass
class ParsedDocument:
    source_file: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def parse_file(file_path: Path, section_dir: Path) -> ParsedDocument:
    """Extract text from a file; `source_file` is its POSIX path relative to
    the section root."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source_file = file_path.relative_to(section_dir).as_posix()
    if file_path.suffix.lower() in KNOWN_BINARY_EXTS:
        return _parse_pdf(file_path, source_file)
    return _parse_text(file_path, source_file)


def _parse_pdf(file_path: Path, source_file: str) -> ParsedDocument:
    import pymupdf

    doc = pymupdf.open(str(file_path))
    pages: list[ParsedPage] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append(ParsedPage(text=text, page_number=page_num))
    finally:
        doc.close()

    logger.info("Parsed PDF %s: %d pages", source_file, len(pages))
    return ParsedDocument(source_file=source_file, pages=pages)


def _parse_text(file_path: Path, source_file: str) -> ParsedDocument:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    logger.info("Parsed %s: %d chars", source_file, len(text))
    return ParsedDocument(
        source_file=source_file,
        pages=[ParsedPage(text=text, page_number=0)],
    )
