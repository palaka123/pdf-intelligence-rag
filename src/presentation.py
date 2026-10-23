"""Display-safety and export helpers for the UI. Pure functions: easy to test.

Model answers and document excerpts are UNTRUSTED (a PDF can contain
injected instructions that shape the answer). Rendering them as Markdown
could, for example, embed an external image that the browser loads
automatically — leaking data without a click. So untrusted text is
escaped before display.
"""

from __future__ import annotations

import re

from src.pdf_processor import ProcessedPDF

# Characters with special meaning in Markdown (and $ for Streamlit's math).
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")

EVIDENCE_LABELS = {
    "explicit": "Stated in the document",
    "inferred": "Inferred from the document",
    "insufficient": "Not found in the document",
}


def escape_markdown(text: str) -> str:
    """Make untrusted text display literally: no links, images, or formatting.

    Single newlines become Markdown line breaks so excerpts keep their lines.
    """
    escaped = _MARKDOWN_SPECIAL.sub(r"\\\1", text)
    return escaped.replace("\n", "  \n")


def build_extracted_text(processed: ProcessedPDF) -> bytes:
    """The extracted text, page by page, as UTF-8 bytes for download."""
    low_text = set(processed.low_text_pages)
    lines = [
        f"Extracted text from {processed.filename}",
        f"Pages: {processed.page_count}",
        "",
    ]
    for page in processed.pages:
        lines.append(f"===== Page {page.page_number} =====")
        if page.page_number in low_text:
            lines.append("(Little or no extractable text — possibly a scanned page.)")
        lines.append(page.text)
        lines.append("")
    # Explicit UTF-8: never rely on the operating system's default encoding.
    return "\n".join(lines).encode("utf-8")


def build_chat_markdown(responses: list, filename: str) -> bytes:
    """The conversation as a Markdown document (UTF-8 bytes) for download.

    `responses` is a list of RAGResponse objects. Untrusted text is escaped
    so the exported file is as safe to open as the app is to view.
    """
    lines = [
        f"# Conversation about {escape_markdown(filename)}",
        "",
        "_Exported from PDF Intelligence. Answers are generated from document "
        "excerpts — verify important facts against the cited pages._",
        "",
    ]
    for number, response in enumerate(responses, start=1):
        answer = response.answer
        lines.append(f"## Q{number}. {escape_markdown(response.question)}")
        lines.append("")
        lines.append(f"**{EVIDENCE_LABELS[answer.evidence_level]}:** {escape_markdown(answer.answer)}")
        if answer.citations:
            sources = ", ".join(f"page {c.page} (`{c.chunk_id}`)" for c in answer.citations)
            lines.append("")
            lines.append(f"**Sources:** {sources}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")