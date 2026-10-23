"""See semantic similarity with real embeddings.

Usage (from the project root):
    python -m scripts.try_embeddings                          # OpenAI (needs .env key)
    python -m scripts.try_embeddings local:bge-small-en-v1.5  # local, free, private
"""

import math
import os
import sys
import time

from dotenv import load_dotenv

from src.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    EmbeddingConfigError,
    check_embeddings,
    create_embeddings,
    get_embedding_spec,
)

QUERY = "How do I reset my password?"
CANDIDATES = [
    "Steps to recover your account login credentials.",
    "Changing the password you use to sign in.",
    "Our office is closed on public holidays.",
    "The recipe needs two cups of flour.",
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """1.0 = same direction (similar meaning); ~0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def main() -> None:
    load_dotenv()  # reads .env into environment variables, locally only
    model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBEDDING_MODEL

    try:
        spec = get_embedding_spec(model_key)
        api_key = os.environ.get("OPENAI_API_KEY") if spec.requires_api_key else None

        start = time.perf_counter()
        embeddings = create_embeddings(model_key, api_key)
        loaded = time.perf_counter()

        check_embeddings(embeddings, model_key)
        query_vector = embeddings.embed_query(QUERY)
        candidate_vectors = embeddings.embed_documents(CANDIDATES)
        done = time.perf_counter()
    except EmbeddingConfigError as exc:
        print(f"Embedding error ({exc.code}): {exc}")
        if exc.code == "unsupported_model":
            print("Available models:", ", ".join(EMBEDDING_MODELS))
        sys.exit(1)

    print(f"Model: {spec.label}")
    print(f"Document text leaves this machine: {'yes' if spec.sends_text_externally else 'no'}")
    print(f"Setup: {loaded - start:.2f}s | Embedding: {done - loaded:.2f}s")
    print(f"Vector length: {len(query_vector)}\n")
    print(f'Query: "{QUERY}"\n')

    ranked = sorted(
        zip(CANDIDATES, candidate_vectors),
        key=lambda pair: cosine_similarity(query_vector, pair[1]),
        reverse=True,
    )
    for text, vector in ranked:
        print(f"{cosine_similarity(query_vector, vector):.3f}  {text}")


if __name__ == "__main__":
    main()