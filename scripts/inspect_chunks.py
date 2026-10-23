"""Inspect how a PDF gets chunked — a tool for tuning size and overlap.

Usage (from the project root):
    python -m scripts.inspect_chunks sample.pdf
    python -m scripts.inspect_chunks sample.pdf 400 60
"""

import sys
from collections import Counter
from pathlib import Path

from src.chunking import chunk_documents
from src.config import CHUNKING
from src.pdf_processor import PDFValidationError, process_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNKING.chunk_size
    overlap = int(sys.argv[3]) if len(sys.argv) > 3 else CHUNKING.chunk_overlap

    try:
        processed = process_pdf(path.read_bytes(), path.name)
        chunks = chunk_documents(processed.to_documents(), size, overlap)
    except (PDFValidationError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if not chunks:
        print("No chunks were produced.")
        sys.exit(1)

    lengths = [c.metadata["char_count"] for c in chunks]
    per_page = Counter(c.metadata["page"] for c in chunks)

    print(f"{processed.filename}: {processed.page_count} pages -> {len(chunks)} chunks "
          f"(size={size}, overlap={overlap})")
    print("Chunks per page:", dict(sorted(per_page.items())))
    print(f"Chunk length: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, "
          f"max {max(lengths)}")

    for chunk in chunks[:3]:
        print(f"\n--- {chunk.metadata['chunk_id']} ({chunk.metadata['char_count']} chars) ---")
        print(chunk.page_content)


if __name__ == "__main__":
    main()