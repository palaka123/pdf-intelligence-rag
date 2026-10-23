"""PDF Intelligence — Streamlit user interface.

Only UI code lives here; all RAG logic is in src/. Streamlit re-runs this
whole script from the top on every interaction, so:
- anything that must survive a rerun lives in st.session_state;
- expensive, user-independent objects (the local model) use st.cache_resource;
- objects holding a user's API key are NEVER cached globally (cache_resource
  is shared by ALL users of the server).
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.config import CHUNKING, PDF_LIMITS, RETRIEVAL
from src.embeddings import EMBEDDING_MODELS, create_embeddings
from src.llm_factory import CHAT_MODELS, DEFAULT_CHAT_MODEL, create_chat_model
from src.pdf_processor import compute_file_hash
from src.presentation import (
    EVIDENCE_LABELS,
    build_chat_markdown,
    build_extracted_text,
    escape_markdown,
)
from src.rag_pipeline import IndexedDocument, RAGResponse, answer_question, index_document

st.set_page_config(page_title="PDF Intelligence", page_icon="📄", layout="wide")

LOCAL_EMBEDDINGS = "local:bge-small-en-v1.5"
EVIDENCE_ICONS = {"explicit": "✅", "inferred": "💡", "insufficient": "⚠️"}


@dataclass(frozen=True)
class Settings:
    """Everything the sidebar collects on this run."""

    uploaded: object | None
    chat_model_key: str
    api_key: str
    embedding_model_key: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    show_inspection: bool


# ------------------------------------------------------------ state & caching

def init_session_state() -> None:
    """Create session keys once; later reruns keep their values."""
    defaults = {"indexed": None, "messages": [], "uploader_version": 0}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner="Loading the local embedding model (first time only)…")
def load_local_embeddings():
    """Loaded once per SERVER and shared by all users — safe because it holds
    no user data and no API key."""
    return create_embeddings(LOCAL_EMBEDDINGS)


def get_embeddings(model_key: str, api_key: str):
    if model_key == LOCAL_EMBEDDINGS:
        return load_local_embeddings()
    # Holds the user's key: created per session, never cached globally.
    return create_embeddings(model_key, api_key)


# ------------------------------------------------------------ sidebar

def render_sidebar() -> Settings:
    with st.sidebar:
        st.title("📄 PDF Intelligence")
        st.caption("Chat with your documents. Every answer cites the pages it comes from.")

        uploaded = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help=(f"Up to {PDF_LIMITS.max_pages} pages and {PDF_LIMITS.max_file_size_mb} MB. "
                  "Text-based PDFs only — scanned pages are not supported (no OCR)."),
            # Changing this key is the only way to empty a file_uploader.
            key=f"uploader_{st.session_state.uploader_version}",
        )

        st.subheader("Models")
        chat_keys = list(CHAT_MODELS)
        chat_model_key = st.selectbox(
            "Answer model",
            chat_keys,
            index=chat_keys.index(DEFAULT_CHAT_MODEL),
            format_func=lambda key: CHAT_MODELS[key].label,
        )
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only during this browser session. Never stored, logged, or shared.",
        ).strip()

        embedding_keys = list(EMBEDDING_MODELS)
        embedding_model_key = st.selectbox(
            "Embedding model",
            embedding_keys,
            index=embedding_keys.index(LOCAL_EMBEDDINGS),
            format_func=lambda key: EMBEDDING_MODELS[key].label,
        )
        if EMBEDDING_MODELS[embedding_model_key].sends_text_externally:
            st.caption("⚠️ Indexing sends the full document text to OpenAI.")
        else:
            st.caption("🔒 Indexing runs on this server; the text isn't sent out for embedding.")
        st.caption("Answering sends the retrieved excerpts to the answer model's provider.")

        with st.expander("Advanced retrieval settings"):
            chunk_size = st.slider("Chunk size (characters)", 300, 1500, CHUNKING.chunk_size, step=50)
            chunk_overlap = st.slider(
                "Chunk overlap (characters)", 0, chunk_size // 2, CHUNKING.chunk_overlap, step=10
            )
            top_k = st.slider("Chunks retrieved per question (top-k)", 1, 8, RETRIEVAL.top_k)
            show_inspection = st.toggle("Show retrieval inspection", value=True)

        if st.button("🗑️ Clear document and chat"):
            st.session_state.indexed = None
            st.session_state.messages = []
            st.session_state.uploader_version += 1  # new key -> empty uploader
            st.rerun()

        st.caption("Clearing removes data from this session. It cannot guarantee "
                   "deletion from hosting-platform logs or caches.")

    return Settings(uploaded, chat_model_key, api_key, embedding_model_key,
                    chunk_size, chunk_overlap, top_k, show_inspection)


# ------------------------------------------------------------ main area

def render_onboarding() -> None:
    st.title("📄 PDF Intelligence — Chat with Your Documents")
    st.markdown(
        f"""
**How it works**

1. **Upload** a text-based PDF (up to {PDF_LIMITS.max_pages} pages) in the sidebar.
2. The app **splits** it into chunks and **indexes** them for meaning-based search.
3. **Ask a question.** The most relevant chunks are retrieved and an AI model
   answers **using only those excerpts**, citing the pages it used.
4. If the document doesn't contain the answer, the app **says so** instead of guessing.

**Good to know:** scanned PDFs (images of text) aren't supported, answers are
only as good as the retrieved excerpts, and you should verify important facts
against the cited pages.
"""
    )


def ensure_index(settings: Settings, file_bytes: bytes) -> IndexedDocument | None:
    """Return a current index, building one only if something relevant changed."""
    file_hash = compute_file_hash(file_bytes)
    indexed: IndexedDocument | None = st.session_state.indexed

    if indexed is not None and indexed.is_current_for(
        file_hash, settings.embedding_model_key, settings.chunk_size, settings.chunk_overlap
    ):
        return indexed  # the common case: no rebuild, no embedding cost

    # A different document means the old conversation no longer applies.
    if indexed is None or indexed.processed.file_hash != file_hash:
        st.session_state.messages = []

    if EMBEDDING_MODELS[settings.embedding_model_key].requires_api_key and not settings.api_key:
        st.info("Enter your OpenAI API key in the sidebar to index with OpenAI "
                "embeddings, or choose the local embedding model.")
        return None

    with st.status("Processing your PDF…", expanded=True) as status:
        st.write("Validating, extracting, chunking and embedding…")
        try:
            embeddings = get_embeddings(settings.embedding_model_key, settings.api_key)
            indexed = index_document(
                file_bytes,
                settings.uploaded.name,
                embeddings,
                settings.embedding_model_key,
                settings.chunk_size,
                settings.chunk_overlap,
            )
        except ValueError as exc:  # all our errors carry a safe message
            status.update(label="Could not process this PDF", state="error")
            st.error(str(exc))
            st.session_state.indexed = None
            return None
        status.update(label=f"Ready — {indexed.index.size} chunks indexed",
                      state="complete", expanded=False)

    st.session_state.indexed = indexed
    return indexed


def render_document_overview(indexed: IndexedDocument, file_bytes: bytes) -> None:
    processed = indexed.processed
    st.subheader(f"📄 {processed.filename}")

    pages_col, chunks_col, skipped_col = st.columns(3)
    pages_col.metric("Pages", processed.page_count)
    chunks_col.metric("Chunks indexed", indexed.index.size)
    skipped_col.metric("Pages without text", len(processed.low_text_pages))

    if processed.low_text_pages:
        skipped = ", ".join(str(p) for p in processed.low_text_pages)
        st.warning(f"Page(s) {skipped} have little or no extractable text (possibly "
                   "scanned) and were skipped. Questions about them can't be answered.")

    with st.expander("Downloads and chunk preview"):
        st.download_button("⬇️ Original PDF", file_bytes,
                           file_name=processed.filename, mime="application/pdf")
        st.download_button("⬇️ Extracted text", build_extracted_text(processed),
                           file_name=f"{processed.filename}.txt", mime="text/plain")
        if st.session_state.messages:
            st.download_button("⬇️ Conversation (Markdown)",
                               build_chat_markdown(st.session_state.messages, processed.filename),
                               file_name=f"{processed.filename}-chat.md", mime="text/markdown")

        st.markdown("**First chunks**")
        for chunk in indexed.index.chunks[:3]:
            meta = chunk.metadata
            st.markdown(f"`{meta['chunk_id']}` · page {meta['page']} · {meta['char_count']} characters")
            st.markdown(escape_markdown(chunk.page_content))


def render_exchange(response: RAGResponse, show_inspection: bool) -> None:
    """Show one question and its verified answer. Untrusted text is escaped."""
    with st.chat_message("user"):
        st.markdown(escape_markdown(response.question))

    answer = response.answer
    with st.chat_message("assistant"):
        level = answer.evidence_level
        st.caption(f"{EVIDENCE_ICONS[level]} {EVIDENCE_LABELS[level]}")
        st.markdown(escape_markdown(answer.answer))

        if answer.citations:
            pages = ", ".join(f"page {p}" for p in answer.cited_pages)
            st.caption(f"Sources: {pages}")
            with st.expander(f"View {len(answer.citations)} supporting excerpt(s)"):
                for citation in answer.citations:
                    st.markdown(f"**Page {citation.page}** · `{citation.chunk_id}`")
                    st.markdown(escape_markdown(citation.excerpt))

        if show_inspection:
            with st.expander("🔍 How this answer was found"):
                cited_ids = {c.chunk_id for c in answer.citations}
                st.caption(
                    f"{CHAT_MODELS[response.chat_model_key].label} · prompt "
                    f"{response.prompt_version}. Scores are similarity values, comparable "
                    "only within one embedding model — not confidence percentages."
                )
                st.dataframe(
                    [
                        {
                            "Rank": doc.metadata["rank"],
                            "Chunk": doc.metadata["chunk_id"],
                            "Page": doc.metadata["page"],
                            "Score": round(doc.metadata["score"], 3),
                            "Cited": "✓" if doc.metadata["chunk_id"] in cited_ids else "",
                        }
                        for doc in response.retrieved
                    ],
                    hide_index=True,
                )
                if answer.flags:
                    st.warning("Guardrail actions: " + escape_markdown(", ".join(answer.flags)))


def handle_question(question: str, indexed: IndexedDocument, settings: Settings) -> None:
    if not settings.api_key:
        st.warning("Enter your OpenAI API key in the sidebar to ask questions.")
        return
    with st.spinner("Searching the document and drafting an answer…"):
        try:
            llm = create_chat_model(settings.chat_model_key, settings.api_key)
            response = answer_question(question, indexed, llm,
                                       settings.chat_model_key, k=settings.top_k)
        except ValueError as exc:  # our errors: safe, user-readable messages
            st.error(str(exc))
            return
        except Exception:  # noqa: BLE001 — never show raw tracebacks to users
            st.error("Something unexpected went wrong. Please try again.")
            return
    st.session_state.messages.append(response)
    st.rerun()  # redraw so the new exchange renders with the rest of the history


def main() -> None:
    init_session_state()
    settings = render_sidebar()

    if settings.uploaded is None:
        render_onboarding()
        return

    file_bytes = settings.uploaded.getvalue()
    indexed = ensure_index(settings, file_bytes)
    if indexed is None:
        return

    render_document_overview(indexed, file_bytes)
    st.divider()

    for response in st.session_state.messages:
        render_exchange(response, settings.show_inspection)

    question = st.chat_input("Ask a question about this document…")
    if question:
        handle_question(question, indexed, settings)


main()