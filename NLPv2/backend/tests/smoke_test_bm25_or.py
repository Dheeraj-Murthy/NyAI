#!/usr/bin/env python3
"""Smoke tests for the BM25 leg's OR query mode (issue #12).

Offline: psycopg2 is replaced by a fake connection that records the SQL and
returns canned rows, so this checks term selection and which query is sent,
not Postgres itself.

Run: cd NLPv2/backend && python tests/smoke_test_bm25_or.py
"""
import os
import sys
import types

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

# The retriever imports these at module level; stub them only if absent.
for _name in ("psycopg2", "sentence_transformers"):
    try:
        __import__(_name)
    except ImportError:
        _stub = types.ModuleType(_name)
        _stub.SentenceTransformer = object
        _stub.connect = None
        sys.modules[_name] = _stub

import retrieval.retriever as retriever_mod  # noqa: E402
from retrieval.retriever import LegalRetriever, _select_or_terms  # noqa: E402

# One result row as the judgment BM25 query returns it.
ROW = ("text", "A", "B", "SC", 1990, "body", 7, 3, 0.4)


class FakeCursor:
    def __init__(self, lexemes, stats):
        self.lexemes, self.stats, self.executed, self._rows = lexemes, stats, [], []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "unnest(to_tsvector" in sql:
            self._rows = [(l,) for l in self.lexemes]
        elif "pg_stats" in sql:
            self._rows = [self.stats] if self.stats else []
        else:
            self._rows = [ROW]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        return self.cur

    def close(self):
        pass


def make_retriever(mode, lexemes, stats):
    cur = FakeCursor(lexemes, stats)
    retriever_mod.psycopg2.connect = lambda dsn: FakeConn(cur)
    r = LegalRetriever.__new__(LegalRetriever)  # skip loading the encoder
    r.db_connection_string, r.bm25_mode = "fake", mode
    return r, cur


def test_select_or_terms():
    print("--- Testing OR term selection ---")
    df = {"court": 0.9, "appeal": 0.3, "bail": 0.02, "grant": 0.04}
    lexemes = ["court", "anticipatori", "bail", "bail", "grant", "appeal", "302a"]
    # Common terms dropped, duplicates removed, unseen (rarest) first in query order.
    assert _select_or_terms(lexemes, df, 0.05, 10) == ["anticipatori", "302a", "bail", "grant"]
    assert _select_or_terms(lexemes, df, 0.05, 2) == ["anticipatori", "302a"]
    assert _select_or_terms(lexemes, {}, 0.05, 10) == ["court", "anticipatori", "bail", "grant", "appeal", "302a"]
    assert _select_or_terms([], df, 0.05, 10) == []
    print("✓ term selection passed")


def test_or_mode_query():
    print("--- Testing OR-mode SQL ---")
    stats = (["bail", "court"], [0.02, 0.9, 0.02, 0.9, 0.0])
    r, cur = make_retriever("or", ["court", "anticipatori", "bail"], stats)
    out = r.retrieve_bm25_candidates("Anticipatory bail; s.438 Cr.P.C. — the Court's view", candidate_k=5)
    assert [c["judgment_id"] for c in out] == [3] and out[0]["bm25_score"] == 0.4
    sql_words, (words,) = cur.executed[0]
    assert "unnest(to_tsvector('english'" in sql_words
    assert words == "Anticipatory bail s 438 Cr P C the Court s view"  # punctuation can't reach to_tsquery
    sql, params = cur.executed[-1]
    assert "to_tsquery('simple'" in sql and "ts_rank_cd" in sql and "plainto_tsquery" not in sql
    assert params == ("anticipatori | bail", 5)
    # Stats are read once per retriever.
    r.retrieve_bm25_candidates("bail", candidate_k=5)
    assert sum("pg_stats" in s for s, _ in cur.executed) == 1
    print("✓ OR-mode SQL passed")


def test_or_mode_nothing_to_search():
    print("--- Testing OR mode with no usable terms ---")
    r, cur = make_retriever("or", [], None)
    assert r.retrieve_bm25_candidates("the of and", candidate_k=5) == []
    r, cur = make_retriever("or", [], None)
    assert r.retrieve_bm25_candidates("—;!", candidate_k=5) == [] and cur.executed == []
    # Every term too common: no query sent rather than matching every chunk.
    r, cur = make_retriever("or", ["court"], (["court"], [0.9, 0.9, 0.9, 0.0]))
    assert r.retrieve_bm25_candidates("court", candidate_k=5) == []
    assert not any("judgment_chunks jc" in s for s, _ in cur.executed)
    print("✓ empty-query handling passed")


def test_and_mode_unchanged():
    print("--- Testing AND mode (old behaviour) ---")
    r, cur = make_retriever("and", [], None)
    out = r.retrieve_bm25_candidates("anticipatory bail", candidate_k=4)
    assert len(out) == 1 and len(cur.executed) == 1
    sql, params = cur.executed[0]
    assert "plainto_tsquery('english'" in sql and "ts_rank(" in sql
    assert params == ("anticipatory bail", 4)
    print("✓ AND mode passed")


def test_mode_validation():
    print("--- Testing bm25_mode validation ---")
    try:
        LegalRetriever(bm25_mode="xor")
    except ValueError:
        pass
    else:
        raise AssertionError("bad bm25_mode accepted")
    assert LegalRetriever.bm25_mode == "or"
    print("✓ validation passed")


if __name__ == "__main__":
    test_select_or_terms()
    test_or_mode_query()
    test_or_mode_nothing_to_search()
    test_and_mode_unchanged()
    test_mode_validation()
    print("\nAll BM25 OR-mode smoke tests passed.")
