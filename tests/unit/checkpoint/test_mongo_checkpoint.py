# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import sys
from types import SimpleNamespace
import pytest
from src.graph.checkpoint import ChatStreamManager


def test_init_without_checkpoint_saver():
    mgr = ChatStreamManager(checkpoint_saver=False, db_uri="")
    assert mgr.checkpoint_saver is False
    # DB connections are not created when saver is disabled
    assert mgr.mongo_client is None
    assert mgr.postgres_conn is None


def test_process_stream_partial_returns_true_and_stores_chunk(monkeypatch):
    # Patch Mongo init to no-op for speed
    def _no_mongo(self):
        self.mongo_client = None
        self.mongo_db = None

    monkeypatch.setattr(ChatStreamManager, "_init_mongodb", _no_mongo, raising=True)

    mgr = ChatStreamManager(
        checkpoint_saver=True,
        db_uri="mongodb://admin:admin@localhost:27017/checkpointing_db?authSource=admin",
    )
    ok = mgr.process_stream_message("t1", "hello", finish_reason="partial")
    assert ok is True
    # Verify the chunk was stored in the in-memory store
    items = mgr.store.search(("messages", "t1"), limit=10)
    values = [it.dict()["value"] for it in items]
    assert "hello" in values


def test_persist_not_attempted_when_saver_disabled():
    mgr = ChatStreamManager(checkpoint_saver=False, db_uri="")
    # stop should try to persist, but saver disabled => returns False
    assert mgr.process_stream_message("t1", "hello", finish_reason="stop") is False


def test_persist_mongodb_called_with_aggregated_chunks(monkeypatch):
    # Avoid real connection by making mongo_db truthy so MongoDB branch is used
    def _fake_mongo(self):
        self.mongo_client = None
        self.mongo_db = object()

    monkeypatch.setattr(ChatStreamManager, "_init_mongodb", _fake_mongo, raising=True)

    mgr = ChatStreamManager(
        checkpoint_saver=True,
        db_uri="mongodb://admin:admin@localhost:27017/checkpointing_db?authSource=admin",
    )

    captured = {}

    def fake_persist(self, thread_id, messages):  # signature must match
        captured["thread_id"] = thread_id
        captured["messages"] = list(messages)
        return True

    monkeypatch.setattr(
        ChatStreamManager, "_persist_to_mongodb", fake_persist, raising=True
    )

    assert mgr.process_stream_message("t3", "Hello", finish_reason="partial") is True
    assert mgr.process_stream_message("t3", " World", finish_reason="stop") is True

    assert captured["thread_id"] == "t3"
    # Order is expected to be chunk_0, chunk_1
    assert captured["messages"] == ["Hello", " World"]


def test_invalid_inputs_return_false(monkeypatch):
    def _no_mongo(self):
        self.mongo_client = None
        self.mongo_db = None

    monkeypatch.setattr(ChatStreamManager, "_init_mongodb", _no_mongo, raising=True)

    mgr = ChatStreamManager(
        checkpoint_saver=True,
        db_uri="mongodb://admin:admin@localhost:27017/checkpointing_db?authSource=admin",
    )
    assert mgr.process_stream_message("", "msg", finish_reason="partial") is False
    assert mgr.process_stream_message("tid", "", finish_reason="partial") is False
