"""End-to-end: index a PDF and ask questions with a real LLM.

Usage (from the project root):
    python -m scripts.ask sample.pdf
    python -m scripts.ask sample.pdf --chat-model openai:gpt-4.1-mini
    python -m scripts.ask sample.pdf --bad-key      # check invalid-key handling

Privacy: embeddings run locally, but the top retrieved excerpts of each
question ARE sent to OpenAI to generate the answer.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import create_embeddings
from src.llm_factory import CHAT_MODELS, DEFAULT_CHAT_MODEL, create_chat_model
from src.rag_pipeline import answer_question, index_document

LOCAL_EMBEDDINGS = "local:bge-small-en-v1.5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, choices=list(CHAT_MODELS))
    parser.add_argument("--bad-key", action="store_true", help="use an invalid API key on purpose")
    args = parser.parse_args()

    load_dotenv()
    api_key = "sk-invalid-key-for-testing" if args.bad_key else os.environ.get("OPENAI_API_KEY")

    try:
        embeddings = create_embeddings(LOCAL_EMBEDDINGS)
        indexed = index_document(args.pdf.read_bytes(), args.pdf.name, embeddings, LOCAL_EMBEDDINGS)
        llm = create_chat_model(args.chat_model, api_key)
    except ValueError as exc:
        print(f"Setup error ({getattr(exc, 'code', 'error')}): {exc}")
        return

    p = indexed.processed
    print(f"Indexed {p.filename}: {p.page_count} pages, {indexed.index.size} chunks. "
          f"Answering with {CHAT_MODELS[args.chat_model].label}.\n")

    while True:
        question = input("Question (Enter to quit): ").strip()
        if not question:
            break
        try:
            response = answer_question(question, indexed, llm, args.chat_model)
        except ValueError as exc:
            print(f"  Error ({getattr(exc, 'code', 'error')}): {exc}\n")
            continue

        a = response.answer
        print(f"  [{a.evidence_level}] {a.answer}")
        for c in a.citations:
            print(f"    cites {c.chunk_id} (page {c.page})")
        retrieved = ", ".join(f"{d.metadata['chunk_id']}:{d.metadata['score']:.2f}" for d in response.retrieved)
        print(f"    retrieved: {retrieved}")
        if a.flags:
            print(f"    guardrail flags: {', '.join(a.flags)}")
        print()


if __name__ == "__main__":
    main()