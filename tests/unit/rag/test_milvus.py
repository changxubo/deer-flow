# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag import milvus as milvus_mod
from src.rag.milvus import MilvusProvider


class _DummyEmbeddings:
    """Deterministic dummy embedding model to avoid external dependencies.
    Mimics the minimal interface needed by the retriever. Always returns the
    same vector (or a provided one) to keep tests deterministic & fast.
    """

    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2, 0.3]

    def embed_query(self, text: str):  # returns deterministic list
        """Return a copy of the static embedding vector."""
        return list(self.vec)


@pytest.fixture(autouse=True)
def base_env(monkeypatch):
    """Provide baseline env vars & patch embedding model for all tests.
    Ensures tests do not make network calls or depend on real Milvus instances.
    """
    monkeypatch.setenv("MILVUS_EMBEDDING_MODEL", "text-embedding-ada-002")
    monkeypatch.setenv("MILVUS_COLLECTION", "cov_collection")
    monkeypatch.setenv("MILVUS_URI", "local.db")  # default lite
    # Ensure embeddings code path is patched (avoid external deps)
    monkeypatch.setattr(milvus_mod, "OpenAIEmbeddings", _DummyEmbeddings, raising=False)
    yield


def _patch_init(monkeypatch):
    """Patch retriever initialization to use dummy embedding model."""
    monkeypatch.setattr(
        MilvusProvider,
        "_init_embedding_model",
        lambda self: setattr(self, "embedding_model", _DummyEmbeddings()),
    )


def test_get_embedding_dimension_explicit(monkeypatch):
    monkeypatch.setenv("MILVUS_EMBEDDING_DIM", "777")
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    assert retriever.embedding_dim == 777


def test_get_embedding_dimension_unknown_model(monkeypatch):
    monkeypatch.delenv("MILVUS_EMBEDDING_DIM", raising=False)
    monkeypatch.setenv("MILVUS_EMBEDDING_MODEL", "unknown-model-x")
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    # falls back to default 1536
    assert retriever.embedding_dim == 1536


def test_is_milvus_lite_variants(monkeypatch):
    _patch_init(monkeypatch)
    monkeypatch.setenv("MILVUS_URI", "mydb.db")
    assert MilvusProvider()._is_milvus_lite() is True
    monkeypatch.setenv("MILVUS_URI", "relative_path_store")
    assert MilvusProvider()._is_milvus_lite() is True
    monkeypatch.setenv("MILVUS_URI", "http://host:19530")
    assert MilvusProvider()._is_milvus_lite() is False


def test_create_collection_lite(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    created: dict = {}

    class DummyMilvusLite:
        def list_collections(self):  # noqa: D401
            return []  # empty triggers creation

        def create_collection(
            self, collection_name, schema, index_params
        ):  # noqa: D401
            created["name"] = collection_name
            created["schema"] = schema
            created["index"] = index_params

    retriever.client = DummyMilvusLite()
    retriever._ensure_collection_exists()
    assert created["name"] == retriever.collection_name


def test_ensure_collection_exists_remote(monkeypatch):
    _patch_init(monkeypatch)
    monkeypatch.setenv("MILVUS_URI", "http://remote:19530")
    retriever = MilvusProvider()
    # remote path, nothing thrown
    retriever.client = SimpleNamespace()
    retriever._ensure_collection_exists()


def test_get_existing_document_ids_lite(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()

    class DummyMilvusLite:
        def query(self, collection_name, filter, output_fields, limit):  # noqa: D401
            return [
                {retriever.id_field: "a"},
                {retriever.id_field: "b"},
                {"other": "ignored"},
            ]

    retriever.client = DummyMilvusLite()
    assert retriever._get_existing_document_ids() == {"a", "b"}


def test_get_existing_document_ids_remote(monkeypatch):
    _patch_init(monkeypatch)
    monkeypatch.setenv("MILVUS_URI", "http://x")
    retriever = MilvusProvider()
    retriever.client = object()
    assert retriever._get_existing_document_ids() == set()


def test_insert_document_chunk_lite_and_error(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()

    captured = {}

    class DummyMilvusLite:
        def insert(self, collection_name, data):  # noqa: D401
            captured["data"] = data

    retriever.client = DummyMilvusLite()
    retriever._insert_document_chunk(
        doc_id="id1", content="hello", title="T", url="u", metadata={"m": 1}
    )
    assert captured["data"][0][retriever.id_field] == "id1"

    # error path: patch embedding to raise
    def bad_embed(text):  # noqa: D401
        raise RuntimeError("boom")

    retriever.embedding_model.embed_query = bad_embed  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        retriever._insert_document_chunk(
            doc_id="id2", content="err", title="T", url="u", metadata={}
        )


def test_insert_document_chunk_remote(monkeypatch):
    _patch_init(monkeypatch)
    monkeypatch.setenv("MILVUS_URI", "http://remote")
    retriever = MilvusProvider()
    added = {}

    class RemoteClient:
        def add_texts(self, texts, metadatas):  # noqa: D401
            added["texts"] = texts
            added["meta"] = metadatas

    retriever.client = RemoteClient()
    retriever._insert_document_chunk(
        doc_id="idx", content="ct", title="Title", url="urlx", metadata={"k": 2}
    )
    assert added["meta"][0][retriever.id_field] == "idx"


def test_connect_lite_and_error(monkeypatch):
    # patch MilvusClient to a dummy
    class FakeMilvusClient:
        def __init__(self, uri):  # noqa: D401
            self.uri = uri

        def list_collections(self):  # noqa: D401
            return []

        def create_collection(self, **kwargs):  # noqa: D401
            pass

    monkeypatch.setattr(milvus_mod, "MilvusClient", FakeMilvusClient)
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    retriever._connect()
    assert isinstance(retriever.client, FakeMilvusClient)

    # error path: patch MilvusClient to raise
    class BadClient:
        def __init__(self, uri):  # noqa: D401
            raise RuntimeError("fail connect")

    monkeypatch.setattr(milvus_mod, "MilvusClient", BadClient)
    retriever2 = MilvusProvider()
    with pytest.raises(ConnectionError):
        retriever2._connect()


def test_connect_remote(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://remote")
    _patch_init(monkeypatch)
    created = {}

    class FakeLangchainMilvus:
        def __init__(self, **kwargs):  # noqa: D401
            created.update(kwargs)

    monkeypatch.setattr(milvus_mod, "LangchainMilvus", FakeLangchainMilvus)
    retriever = MilvusProvider()
    retriever._connect()
    assert created["collection_name"] == retriever.collection_name


def test_list_resources_fallback_on_connect_failure(monkeypatch, tmp_path):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()

    # create local examples dir with one file
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "sample.md").write_text("# Title\n\nBody", encoding="utf-8")
    monkeypatch.setenv("MILVUS_EXAMPLES_DIR", str(examples_dir.name))
    retriever.examples_dir = examples_dir.name

    # force connect failure via generator throw
    monkeypatch.setattr(
        retriever, "_connect", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    res = retriever.list_resources()
    assert res and res[0].title == "Title"


def test_list_resources_remote_failure(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://remote")
    _patch_init(monkeypatch)
    retriever = MilvusProvider()

    # Provide minimal working local examples dir (none -> returns [])
    monkeypatch.setattr(retriever, "_list_local_markdown_resources", lambda: [])

    # patch client to raise inside similarity_search to trigger fallback path
    class BadClient:
        def similarity_search(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("fail")

    retriever.client = BadClient()
    # Should fallback to [] without raising
    assert retriever.list_resources() == []


def test_list_local_markdown_resources_empty(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    monkeypatch.setenv("MILVUS_EXAMPLES_DIR", "nonexistent_dir")
    retriever.examples_dir = "nonexistent_dir"
    assert retriever._list_local_markdown_resources() == []


def test_query_relevant_documents_error(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    retriever.embedding_model.embed_query = lambda text: (  # type: ignore
        _ for _ in ()
    ).throw(RuntimeError("embed fail"))
    with pytest.raises(RuntimeError):
        retriever.query_relevant_documents("q")


def test_create_collection_when_client_exists(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    retriever.client = SimpleNamespace(closed=False)
    # remote vs lite path difference handled by _is_milvus_lite
    retriever.create_collection()  # should no-op gracefully


def test_load_examples_force_reload(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    retriever.client = SimpleNamespace()
    called = {"clear": 0, "load": 0}
    monkeypatch.setattr(
        retriever, "_clear_example_documents", lambda: called.__setitem__("clear", 1)
    )
    monkeypatch.setattr(
        retriever, "_load_example_files", lambda: called.__setitem__("load", 1)
    )
    retriever.load_examples(force_reload=True)
    assert called == {"clear": 1, "load": 1}


def test_clear_example_documents_remote(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://remote")
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    retriever.client = SimpleNamespace()
    # Should just log and not raise
    retriever._clear_example_documents()


def test_clear_example_documents_lite(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    deleted = {}

    class DummyMilvusLite:
        def query(self, **kwargs):  # noqa: D401
            return [
                {retriever.id_field: "ex1"},
                {retriever.id_field: "ex2"},
            ]

        def delete(self, collection_name, ids):  # noqa: D401
            deleted["ids"] = ids

    retriever.client = DummyMilvusLite()
    retriever._clear_example_documents()
    assert deleted["ids"] == ["ex1", "ex2"]


def test_get_loaded_examples_lite_and_error(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()

    class DummyMilvusLite:
        def query(self, **kwargs):  # noqa: D401
            return [
                {
                    retriever.id_field: "id1",
                    retriever.title_field: "T1",
                    retriever.url_field: "u1",
                    "file": "f1",
                }
            ]

    retriever.client = DummyMilvusLite()
    loaded = retriever.get_loaded_examples()
    assert loaded[0]["id"] == "id1"

    # error path
    class BadClient:
        def query(self, **kwargs):  # noqa: D401
            raise RuntimeError("fail")

    retriever.client = BadClient()
    assert retriever.get_loaded_examples() == []


def test_get_loaded_examples_remote(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://remote")
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    retriever.client = SimpleNamespace()
    assert retriever.get_loaded_examples() == []


def test_close_lite_and_remote(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    closed = {"c": 0}

    class DummyMilvusLite:
        def close(self):  # noqa: D401
            closed["c"] += 1

        def list_collections(self):  # noqa: D401
            return []

        def create_collection(self, **kwargs):  # noqa: D401
            pass

    retriever.client = DummyMilvusLite()
    retriever.close()
    assert closed["c"] == 1

    # remote path: no close attr usage expected
    monkeypatch.setenv("MILVUS_URI", "http://remote")
    retriever2 = MilvusProvider()
    retriever2.client = SimpleNamespace()
    retriever2.close()  # should not raise


def test_module_level_load_examples(monkeypatch):
    # Case where rag provider not milvus -> no call
    called = {"inst": 0}

    class DummyProv(MilvusProvider):  # type: ignore
        def load_examples(self, *args, **kwargs):  # noqa: D401
            called["inst"] += 1

    monkeypatch.setattr(milvus_mod, "MilvusProvider", DummyProv)
    monkeypatch.setenv("RAG_PROVIDER", "other")
    monkeypatch.setenv("MILVUS_AUTO_LOAD_EXAMPLES", "1")
    milvus_mod.load_examples()
    assert called["inst"] == 0

    # Now provider matches and auto load true
    monkeypatch.setenv("RAG_PROVIDER", "milvus")
    milvus_mod.load_examples()
    assert called["inst"] == 1


def test_get_embedding_invalid_output(monkeypatch):
    _patch_init(monkeypatch)
    retriever = MilvusProvider()
    # patch embedding model to return invalid output (empty list)
    retriever.embedding_model.embed_query = lambda text: []  # type: ignore
    with pytest.raises(RuntimeError):
        retriever._get_embedding("text")
