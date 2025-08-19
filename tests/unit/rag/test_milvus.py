# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os

import pytest

from src.rag.milvus import MilvusRetriever, Resource, Chunk, Document


class DummyEmbeddingModel:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_query(self, text: str):  # LangChain OpenAIEmbeddings interface subset
        # return deterministic vector length dim
        return [float(len(text) % 7 + i) for i in range(self.dim)]


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    """Provide minimal environment variables & ensure deterministic state."""
    monkeypatch.setenv("MILVUS_EMBEDDING_MODEL", "text-embedding-ada-002")
    monkeypatch.setenv("MILVUS_TOP_K", "5")
    monkeypatch.setenv("MILVUS_COLLECTION", "test_collection")
    # Force using a lite URI path to exercise that branch by default
    monkeypatch.setenv("MILVUS_URI", "local.db")
    yield


@pytest.fixture
def lite_retriever(monkeypatch):
    """MilvusRetriever configured for Milvus Lite with a fake client."""

    # Patch embedding init to avoid external imports
    monkeypatch.setattr(
        MilvusRetriever,
        "_init_embedding_model",
        lambda self: setattr(self, "embedding_model", DummyEmbeddingModel()),
    )
    # Force tiny chunk size for easier splitting tests
    monkeypatch.setenv("MILVUS_CHUNK_SIZE", "50")

    r = MilvusRetriever()

    # Fake lite client implementing subset of operations
    class FakeLiteClient:
        def __init__(self):
            self.inserted = []
            self.collections = {r.collection_name}

        # Collection ops
        def list_collections(self):
            return list(self.collections)

        def create_collection(self, **kwargs):  # pragma: no cover (not used normally)
            self.collections.add(kwargs.get("collection_name"))

        # Data ops
        def insert(self, collection_name, data):
            self.inserted.extend(data)

        def query(self, collection_name, filter, output_fields, limit):
            # Return only 'example' docs if filter matches
            if "source == 'examples'" in filter:
                return [
                    {
                        r.id_field: "example_doc1",
                        r.title_field: "Example Doc 1",
                        r.url_field: "milvus://test_collection/doc1.md",
                    }
                ]
            return []

        def search(
            self, collection_name, data, anns_field, param, limit, output_fields
        ):
            # Return a single hit shaped like pymilvus result
            return [
                [
                    {
                        "entity": {
                            r.id_field: "docA",
                            r.content_field: "Relevant content A",
                            r.title_field: "Title A",
                            r.url_field: "milvus://test_collection/docA",
                        },
                        "distance": 0.87,
                    }
                ]
            ]

        def delete(self, collection_name, ids):  # pragma: no cover
            pass

        def close(self):  # pragma: no cover
            pass

    r.client = FakeLiteClient()
    # Avoid re-building client
    monkeypatch.setattr(r, "_connect", lambda: None)
    return r


@pytest.fixture
def remote_retriever(monkeypatch):
    """MilvusRetriever configured for remote (LangChain) path with fake client."""
    monkeypatch.setenv("MILVUS_URI", "http://remote-milvus:19530")
    monkeypatch.setattr(
        MilvusRetriever,
        "_init_embedding_model",
        lambda self: setattr(self, "embedding_model", DummyEmbeddingModel()),
    )
    r = MilvusRetriever()

    class FakeDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    class FakeRemoteClient:
        def similarity_search(self, query, k, expr):
            return [
                FakeDoc(
                    "Example page",  # page_content
                    {
                        r.id_field: "ex1",
                        r.title_field: "Example 1",
                        r.url_field: "milvus://test_collection/ex1.md",
                    },
                )
            ]

        def similarity_search_with_score(self, query, k):
            return [
                (
                    FakeDoc(
                        "Remote content",  # page_content
                        {
                            r.id_field: "rid1",
                            r.title_field: "Remote Title",
                            r.url_field: "milvus://test_collection/rid1",
                        },
                    ),
                    0.42,
                )
            ]

        def add_texts(self, texts, metadatas):  # pragma: no cover
            pass

    r.client = FakeRemoteClient()
    monkeypatch.setattr(r, "_connect", lambda: None)
    return r


def test_split_content_chunking(lite_retriever):
    content = "Paragraph1." + "a" * 30 + "\n\n" + "Paragraph2." + "b" * 30
    chunks = lite_retriever._split_content(content)
    # Because chunk size = 50, likely split into >=2 chunks
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)  # allow small overhead


def test_generate_doc_id_stability(tmp_path, lite_retriever):
    p = tmp_path / "file.md"
    p.write_text("hello")
    id1 = lite_retriever._generate_doc_id(p)
    id2 = lite_retriever._generate_doc_id(p)
    assert id1 == id2  # stable for unchanged file


def test_get_embedding_success(lite_retriever):
    emb = lite_retriever._get_embedding("hello world")
    assert isinstance(emb, list) and emb


@pytest.mark.parametrize("bad", [None, 123, "   "])
def test_get_embedding_invalid_inputs_raise(lite_retriever, bad):
    with pytest.raises(RuntimeError):
        lite_retriever._get_embedding(bad)  # type: ignore[arg-type]


def test_list_resources_lite(lite_retriever):
    resources = lite_retriever.list_resources()
    assert resources and isinstance(resources[0], Resource)
    assert resources[0].uri.startswith("milvus://")


def test_query_relevant_documents_lite(lite_retriever):
    docs = lite_retriever.query_relevant_documents("test query")
    assert len(docs) == 1
    d = docs[0]
    assert isinstance(d, Document)
    assert d.chunks and isinstance(d.chunks[0], Chunk)
    assert d.chunks[0].similarity > 0


def test_query_relevant_documents_lite_with_resource_filter(lite_retriever):
    # Resource not matching doc will filter out results
    res = [Resource(uri="milvus://test_collection/other.md", title="Other")]  # no match
    docs = lite_retriever.query_relevant_documents("query", res)
    assert docs == []

    res2 = [Resource(uri="milvus://test_collection/docA", title="Doc A")]
    docs2 = lite_retriever.query_relevant_documents("query", res2)
    assert docs2 and docs2[0].id == "docA"


def test_list_resources_remote(remote_retriever):
    resources = remote_retriever.list_resources()
    assert len(resources) == 1
    assert resources[0].title == "Example 1"


def test_query_relevant_documents_remote(remote_retriever):
    docs = remote_retriever.query_relevant_documents("some query")
    assert len(docs) == 1
    assert docs[0].chunks[0].similarity == 0.42


def test_query_relevant_documents_remote_with_filter(remote_retriever):
    # Provide resource that does not match -> filtered out
    res = [Resource(uri="milvus://test_collection/other", title="Other")]
    docs = remote_retriever.query_relevant_documents("q", res)
    assert docs == []

    # Matching resource
    res2 = [Resource(uri="milvus://test_collection/rid1", title="Match")]
    docs2 = remote_retriever.query_relevant_documents("q", res2)
    assert docs2 and docs2[0].id == "rid1"


def test_load_examples_invokes_insert(monkeypatch, lite_retriever):
    # Patch helpers to avoid real embedding + limit scope to a single artificial file
    inserted = []
    monkeypatch.setattr(
        lite_retriever,
        "_get_existing_document_ids",
        lambda: set(),
    )
    monkeypatch.setattr(
        lite_retriever,
        "_insert_document_chunk",
        lambda **kwargs: inserted.append(kwargs),
    )

    # Point examples dir to real repo examples (already present) or create one temp file
    # Prefer creating a temp examples directory with single file for deterministic count
    temp_examples = os.path.join(os.getcwd(), "temp_examples")
    os.makedirs(temp_examples, exist_ok=True)
    sample = os.path.join(temp_examples, "sample.md")
    with open(sample, "w", encoding="utf-8") as f:
        f.write("# Sample Title\n\nContent paragraph one.\n\nSecond paragraph.")
    monkeypatch.setenv("MILVUS_EXAMPLES_DIR", "temp_examples")

    lite_retriever.examples_dir = "temp_examples"  # reflect override
    lite_retriever.load_examples()
    assert inserted, "Expected at least one chunk inserted from example file"
    # Clean up temp examples directory contents (not directory) for idempotent reruns
    for fn in os.listdir(temp_examples):  # pragma: no cover (cleanup)
        try:
            os.remove(os.path.join(temp_examples, fn))
        except OSError:
            pass


def test_invalid_embedding_provider(monkeypatch):
    monkeypatch.setenv("MILVUS_EMBEDDING_PROVIDER", "invalid")
    monkeypatch.setattr("src.rag.milvus.EMBEDDINGS_AVAILABLE", True, raising=False)
    # Patch OpenAIEmbeddings symbol expected by init to a dummy so import path works
    monkeypatch.setattr(
        "src.rag.milvus.OpenAIEmbeddings", DummyEmbeddingModel, raising=False
    )
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        MilvusRetriever()
