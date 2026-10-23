"""Ask questions against a real PDF and see exactly what retrieval returns.

Usage (from the project root):
    python -m scripts.try_search sample.pdf                                # local model (private)
    python -m scripts.try_search sample.pdf openai:text-embedding-3-small  # sends text to OpenAI
Type a question at the prompt; press Enter on an empty line to quit.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import chunk_documents
from src.config import RETRIEVAL
from src.embeddings import EmbeddingConfigError, create_embeddings, get_embedding_spec
from src.pdf_processor import process_pdf
from src.vector_store import DocumentIndex, VectorStoreError


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    model_key = sys.argv[2] if len(sys.argv) > 2 else "local:bge-small-en-v1.5"

    try:
        spec = get_embedding_spec(model_key)
        processed = process_pdf(path.read_bytes(), path.name)
        chunks = chunk_documents(processed.to_documents())
        api_key = os.environ.get("OPENAI_API_KEY") if spec.requires_api_key else None
        embeddings = create_embeddings(model_key, api_key)

        start = time.perf_counter()
        index = DocumentIndex.build(chunks, embeddings, model_key, processed.file_hash)
        elapsed = time.perf_counter() - start
    except ValueError as exc:  # all our error types subclass ValueError
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Indexed {index.size} chunks from {processed.filename} "
          f"with {spec.label} in {elapsed:.2f}s")
    print(f"Document text sent to an external API: "
          f"{'yes' if spec.sends_text_externally else 'no'}\n")

    while True:
        question = input("Question (Enter to quit): ").strip()
        if not question:
            break
        try:
            results = index.search(question, k=RETRIEVAL.top_k)
        except (EmbeddingConfigError, VectorStoreError) as exc:
            print(f"Error: {exc}\n")
            continue
        for r in results:
            preview = r.chunk.page_content.replace("\n", " ")[:100]
            print(f"  #{r.rank}  {r.chunk_id:<7} page {r.page}  score {r.score:.3f}  | {preview}")
        print()


if __name__ == "__main__":
    main()