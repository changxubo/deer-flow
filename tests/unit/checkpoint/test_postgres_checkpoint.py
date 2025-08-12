# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import sys
import types
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
    # Patch Postgres init to no-op
    def _no_pg(self):
        self.postgres_conn = None

    monkeypatch.setattr(ChatStreamManager, "_init_postgresql", _no_pg, raising=True)

    mgr = ChatStreamManager(
        checkpoint_saver=True,
        db_uri="postgresql://postgres:postgres@localhost:5432/checkpointing_db",
    )
    ok = mgr.process_stream_message("t1", "hello", finish_reason="partial")
    assert ok is True

    # Verify the chunk was stored in the in-memory store
    items = mgr.store.search(("messages", "t1"), limit=10)
    values = [it.dict()["value"] for it in items]
    assert "hello" in values


def test_persist_not_attempted_when_saver_disabled():
    mgr = ChatStreamManager(
        checkpoint_saver=False,
        db_uri="",
    )
    # stop should try to persist, but saver disabled => returns False
    assert mgr.process_stream_message("t1", "hello", finish_reason="stop") is False


def test_persist_postgresql_called_with_aggregated_chunks(monkeypatch):
    # Avoid real connection by making postgres_conn truthy so PostgreSQL branch is used
    def _fake_pg(self):
        class _DummyConn:
            pass

        self.postgres_conn = _DummyConn()

    monkeypatch.setattr(ChatStreamManager, "_init_postgresql", _fake_pg, raising=True)

    mgr = ChatStreamManager(
        checkpoint_saver=True,
        db_uri="postgresql://postgres:postgres@localhost:5432/checkpointing_db",
    )

    captured = {}

    def fake_persist(self, thread_id, messages):  # signature must match
        captured["thread_id"] = thread_id
        captured["messages"] = list(messages)
        return True

    monkeypatch.setattr(
        ChatStreamManager, "_persist_to_postgresql", fake_persist, raising=True
    )

    assert mgr.process_stream_message("t3", "Hello", finish_reason="partial") is True
    assert mgr.process_stream_message("t3", " World", finish_reason="stop") is True

    assert captured["thread_id"] == "t3"
    # Order is expected to be chunk_0, chunk_1
    assert captured["messages"] == ["Hello", " World"]


def test_invalid_inputs_return_false(monkeypatch):
    def _no_pg(self):
        self.postgres_conn = None

    monkeypatch.setattr(ChatStreamManager, "_init_postgresql", _no_pg, raising=True)

    mgr = ChatStreamManager(
        checkpoint_saver=True,
        db_uri="postgresql://postgres:postgres@localhost:5432/checkpointing_db",
    )
    assert mgr.process_stream_message("", "msg", finish_reason="partial") is False
    assert mgr.process_stream_message("tid", "", finish_reason="partial") is False
