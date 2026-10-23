"""Tests for the local FastEmbed option — offline, no model download.

monkeypatch temporarily replaces fastembed.TextEmbedding with a fake for the
duration of one test, then restores the real one automatically.
"""

import sys

import numpy as np
import pytest

from src.embeddings import (
    EMBEDDING_MODELS,
    EmbeddingConfigError,
    check_embeddings,
    create_embeddings,
)

LOCAL = "local:bge-small-en-v1.5"


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding and records which method ran."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.calls = []

    def passage_embed(self, texts):
        self.calls.append("passage")
        for _ in texts:
            yield np.full(384, 0.5, dtype=np.float32)

    def query_embed(self, query):
        self.calls.append("query")
        yield np.full(384, 0.25, dtype=np.float32)


class BrokenTextEmbedding:
    """Simulates a model download failure (e.g. no internet)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("download failed")


@pytest.fixture
def fake_fastembed(monkeypatch):
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)


def test_local_spec_needs_no_key_and_keeps_text_private():
    spec = EMBEDDING_MODELS[LOCAL]
    assert spec.requires_api_key is False
    assert spec.sends_text_externally is False
    assert spec.dimensions == 384


def test_local_embeddings_created_without_api_key(fake_fastembed):
    embeddings = create_embeddings(LOCAL, api_key=None)
    check_embeddings(embeddings, LOCAL)  # 384 dims matches the catalog


def test_documents_use_passage_and_query_uses_query(fake_fastembed):
    embeddings = create_embeddings(LOCAL)

    doc_vectors = embeddings.embed_documents(["chunk one", "chunk two"])
    query_vector = embeddings.embed_query("a question")

    assert len(doc_vectors) == 2
    assert isinstance(doc_vectors[0][0], float)   # plain floats, not numpy
    assert len(query_vector) == 384
    assert embeddings._model.calls == ["passage", "query"]


def test_missing_fastembed_package_gives_clear_error(monkeypatch):
    # A None entry in sys.modules makes `import fastembed` fail.
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(EmbeddingConfigError) as err:
        create_embeddings(LOCAL)
    assert err.value.code == "dependency_missing"


def test_model_download_failure_gives_clear_error(monkeypatch):
    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", BrokenTextEmbedding)
    with pytest.raises(EmbeddingConfigError) as err:
        create_embeddings(LOCAL)
    assert err.value.code == "model_unavailable"