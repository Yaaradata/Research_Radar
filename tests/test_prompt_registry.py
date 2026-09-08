"""Prompt registry integrity tests (no DB required for hash collision logic)."""

from __future__ import annotations

import pytest

from research_radar.prompt_registry import PromptRegistryError, register_prompt, sha256_body


class _FakeConn:
    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}

    def execute(self, sql, params=None):
        sql_l = " ".join(sql.split()).lower()
        if sql_l.startswith("select"):
            kind, version = params
            row = self.rows.get((kind, version))

            class R:
                def fetchone(self_inner):
                    return row

            return R()
        if sql_l.startswith("insert"):
            kind, version, model_name, digest, body = params
            row = {
                "prompt_id": len(self.rows) + 1,
                "kind": kind,
                "version": version,
                "model_name": model_name,
                "body_sha256": digest,
                "body": body,
                "active": True,
                "created_at": None,
            }
            self.rows[(kind, version)] = row

            class R:
                def fetchone(self_inner):
                    return row

            return R()
        raise AssertionError(f"unexpected sql: {sql}")


def test_sha256_stable():
    assert sha256_body("hello") == sha256_body("hello")
    assert sha256_body("hello") != sha256_body("hellp")


def test_register_prompt_idempotent_same_hash():
    conn = _FakeConn()
    a = register_prompt(conn, "screen", "v1", "BODY", model_name="m")
    b = register_prompt(conn, "screen", "v1", "BODY", model_name="m")
    assert a["body_sha256"] == b["body_sha256"]
    assert len(conn.rows) == 1


def test_register_prompt_rejects_silent_body_change():
    conn = _FakeConn()
    register_prompt(conn, "screen", "v1", "BODY A")
    with pytest.raises(PromptRegistryError):
        register_prompt(conn, "screen", "v1", "BODY B")
