"""Run the evaluation set against real models and report the scores.

Usage (from the project root):
    python -m scripts.run_eval
    python -m scripts.run_eval --chat-model openai:gpt-4.1-mini --repeats 3

Cost: about questions x repeats small API calls (well under one US cent
for the default set). Embeddings run locally.
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import create_embeddings
from src.evaluation import format_summary_markdown, load_eval_set, score_response, summarize
from src.llm_factory import CHAT_MODELS, DEFAULT_CHAT_MODEL, create_chat_model
from src.rag_pipeline import answer_question, index_document

EVAL_SET = Path("eval/eval_set.json")
LOCAL_EMBEDDINGS = "local:bge-small-en-v1.5"


def mark(value) -> str:
    return "-" if value is None else ("PASS" if value else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, choices=list(CHAT_MODELS))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    load_dotenv()
    eval_set = load_eval_set(EVAL_SET)
    pdf_path = Path(eval_set.document)
    if not pdf_path.exists():
        raise SystemExit(f"{pdf_path} not found. Run: python -m scripts.make_eval_pdf")

    embeddings = create_embeddings(LOCAL_EMBEDDINGS)
    indexed = index_document(pdf_path.read_bytes(), pdf_path.name, embeddings, LOCAL_EMBEDDINGS,
                             eval_set.chunk_size, eval_set.chunk_overlap)
    llm = create_chat_model(args.chat_model, os.environ.get("OPENAI_API_KEY"))
    print(f"Indexed {indexed.index.size} chunks. Model: {CHAT_MODELS[args.chat_model].label}. "
          f"{len(eval_set.cases)} questions x {args.repeats} runs.\n")

    results = []
    for case in eval_set.cases:
        for run in range(1, args.repeats + 1):
            response = answer_question(case.question, indexed, llm, args.chat_model, k=eval_set.top_k)
            result = score_response(case, response, eval_set.canary, run)
            results.append(result)
            leak = "  CANARY LEAKED!" if result.canary_leaked else ""
            print(f"{case.case_id:<24} run {run}  {result.evidence_level:<12} "
                  f"retrieval={mark(result.retrieval_hit):<4} abstain={mark(result.abstention_correct):<4} "
                  f"cites={mark(result.citations_correct):<4} facts={mark(result.facts_found):<4}{leak}")

    summary = summarize(results)
    title = f"{CHAT_MODELS[args.chat_model].label} - {date.today().isoformat()}"
    markdown = format_summary_markdown(summary, title)
    print("\n" + markdown)

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.chat_model.replace(":", "_")
    (out_dir / f"{slug}.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{slug}.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\nSaved to {out_dir / slug}.json and .md")


if __name__ == "__main__":
    main()