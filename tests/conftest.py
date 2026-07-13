"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown with code blocks, headings, and tables."""
    return (
        "# Main Title\n\n"
        "Introduction paragraph with some text.\n\n"
        "## Code Example\n\n"
        "Here is some code:\n\n"
        "```python\n"
        "def cuda_malloc(size):\n"
        "    ptr = allocate(size)\n"
        "    return ptr\n"
        "```\n\n"
        "## Data Table\n\n"
        "| Function | Returns | Description |\n"
        "| --- | --- | --- |\n"
        "| cudaMalloc | cudaError_t | Allocates device memory |\n"
        "| cudaFree | cudaError_t | Frees device memory |\n\n"
        "## Conclusion\n\n"
        "Final paragraph of the document.\n"
    )


@pytest.fixture
def sample_text() -> str:
    """Sample plain text."""
    return (
        "This is a plain text document about CUDA programming.\n"
        "CUDA is a parallel computing platform by NVIDIA.\n"
        "It allows developers to use GPUs for general purpose processing.\n"
    )
