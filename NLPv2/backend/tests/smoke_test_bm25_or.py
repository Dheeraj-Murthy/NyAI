#!/usr/bin/env python3
"""Smoke tests for the BM25 leg's query modes and the case lookup (issue #12).

Offline: psycopg2 is replaced by a fake connection that records the SQL and
returns canned rows, so this checks term selection, which queries are sent,
the timeout fallback and lookup pinning — not Postgres itself.

Run: cd NLPv2/backend && python tests/smoke_test_bm25_or.py
"""
import os
import sys
import types

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "benchmark"))

# The retriever and reranker import these at module level; stub them only if absent.
for _name in ("psycopg2", "sentence_transformers"):
    try:
        __import__(_name)
    except ImportError:
        _stub = types.ModuleType(_name)
        _stub.SentenceTransformer = object
        _stub.CrossEncoder = object
        _stub.Error = type("Error", (Exception,), {})
        _stub.connect = None
        sys.modules[_name] = _stub

import retrieval.retriever as retriever_mod  # noqa: E402
from judgment_eval import BM25_MODES as EVAL_BM25_MODES  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
from retrieval.retriever import (  # noqa: E402
    BM25_MODES,
    LegalRetriever,
    _select_or_terms,
    lookup_strings,
    named_case_phrases,
)


def row(chunk_id, judgment_id, score=0.4):
    """One result row as the judgment BM25 query returns it."""
    return ("text", "A", "B", "SC", 1990, "body", chunk_id, judgment_id, score)


class Timeout(sys.modules["psycopg2"].Error):
    pgcode = "57014"


class FakeCursor:
    def __init__(self, lexemes, stats, and_rows, or_rows, or_error=None, alias_ids=()):
        self.lexemes, self.stats = lexemes, stats
        self.and_rows, self.or_rows, self.or_error = and_rows, or_rows, or_error
        self.alias_ids = list(alias_ids)
        self.executed, self._rows = [], []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "unnest(to_tsvector" in sql:
            self._rows = [(l,) for l in self.lexemes]
        elif "pg_stats" in sql:
            self._rows = [self.stats] if self.stats else []
        elif "statement_timeout" in sql:
            self._rows = []
        elif "plainto_tsquery" in sql:
            self._rows = list(self.and_rows)
        elif "to_tsquery('simple'" in sql:
            if self.or_error:
                raise self.or_error
            self._rows = list(self.or_rows)
        elif "judgment_aliases" in sql:
            self._rows = [(j,) for j in self.alias_ids]
        elif "DISTINCT ON (jc.judgment_id)" in sql:
            self._rows = [("pinned text", "P", "R", "SC", 1978, "ratio", 900 + j, j, 0.7) for j in params[1]]
        else:
            raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, cur):
        self.cur, self.rolled_back = cur, 0

    def cursor(self):
        return self.cur

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        pass


def make_retriever(mode, cur, graph=None, case_lookup=True):
    conn = FakeConn(cur)
    retriever_mod.psycopg2.connect = lambda dsn: conn
    r = LegalRetriever.__new__(LegalRetriever)  # skip loading the encoder
    r.db_connection_string, r.bm25_mode, r.graph_manager = "fake", mode, graph
    r.case_lookup, r.bm25_stats, r._chunk_doc_freq = case_lookup, retriever_mod.Counter(), None
    r.get_query_embedding = lambda q: [0.1, 0.2]
    return r, conn


def bm25_sql(cur):
    return [(s, p) for s, p in cur.executed if "judgment_chunks jc" in s and "DISTINCT ON" not in s]


def test_modes_and_defaults():
    print("--- Testing modes and defaults ---")
    assert BM25_MODES == EVAL_BM25_MODES == ("and", "or", "and_or")
    assert LegalRetriever.bm25_mode == "and_or" and LegalRetriever.case_lookup is True
    try:
        LegalRetriever(bm25_mode="xor")
    except ValueError:
        pass
    else:
        raise AssertionError("bad bm25_mode accepted")
    print("✓ modes passed")


def test_select_or_terms():
    print("--- Testing OR term selection ---")
    df = {"court": 0.9, "appeal": 0.3, "bail": 0.02, "grant": 0.04}
    lexemes = ["court", "anticipatori", "bail", "bail", "grant", "appeal", "302a"]
    assert _select_or_terms(lexemes, df, 0.05, 10) == ["anticipatori", "302a", "bail", "grant"]
    assert _select_or_terms(lexemes, df, 0.01, 10) == ["anticipatori", "302a"]
    assert _select_or_terms(lexemes, df, 0.05, 2) == ["anticipatori", "302a"]
    assert _select_or_terms([], df, 0.05, 10) == []
    print("✓ term selection passed")


STATS = (["bail", "court"], [0.02, 0.9, 0.02, 0.9, 0.0])


def test_and_mode():
    print("--- Testing AND mode ---")
    cur = FakeCursor([], None, [row(7, 3)], [row(8, 4)])
    r, _ = make_retriever("and", cur)
    out = r.retrieve_bm25_candidates("anticipatory bail", candidate_k=4)
    assert [c["chunk_id"] for c in out] == [7]
    (sql, params), = bm25_sql(cur)
    assert "plainto_tsquery('english'" in sql and "ts_rank(" in sql
    assert params == ("anticipatory bail", [], 4)
    print("✓ AND mode passed")


def test_or_mode():
    print("--- Testing OR mode ---")
    cur = FakeCursor(["court", "anticipatori", "bail"], STATS, [], [row(7, 3)])
    r, _ = make_retriever("or", cur)
    out = r.retrieve_bm25_candidates("Anticipatory bail; s.438 Cr.P.C. — the Court's view", candidate_k=5)
    assert [c["judgment_id"] for c in out] == [3]
    assert cur.executed[0][1] == ("Anticipatory bail s 438 Cr P C the Court s view",)  # no punctuation reaches to_tsquery
    (sql, params), = bm25_sql(cur)
    assert "to_tsquery('simple'" in sql and "ts_rank_cd" in sql
    assert params == ("anticipatori | bail", [], 5)
    # No usable terms: no query at all rather than matching every chunk.
    cur = FakeCursor(["court"], (["court"], [0.9, 0.9, 0.9, 0.0]), [], [row(1, 1)])
    r, _ = make_retriever("or", cur)
    assert r.retrieve_bm25_candidates("court", candidate_k=5) == [] and not bm25_sql(cur)
    print("✓ OR mode passed")


def test_and_or_fill():
    print("--- Testing and_or: AND first, then a narrow OR fill ---")
    stats = (["bail", "court", "anticipatori"], [0.005, 0.9, 0.02, 0.9, 0.0, 0.0])
    cur = FakeCursor(["court", "anticipatori", "bail", "dowri"], stats, [row(7, 3, 0.9)], [row(8, 4), row(9, 5)])
    r, _ = make_retriever("and_or", cur)
    out = r.retrieve_bm25_candidates("anticipatory bail dowry court", candidate_k=3)
    assert [c["chunk_id"] for c in out] == [7, 8, 9]  # AND rows keep the top
    (and_sql, and_p), (or_sql, or_p) = bm25_sql(cur)
    assert "plainto_tsquery" in and_sql and "to_tsquery('simple'" in or_sql
    # 1% cut drops anticipatori (2%) and court; already-found chunk 7 skipped; only 2 slots left.
    assert or_p == ("dowri | bail", [7], 2)
    assert any("SET LOCAL statement_timeout = 500" in s for s, _ in cur.executed)
    assert r.bm25_stats == {"calls": 1, "fills": 1}

    # AND already full: no fill query.
    cur = FakeCursor(["bail"], STATS, [row(1, 1), row(2, 2)], [row(8, 4)])
    r, _ = make_retriever("and_or", cur)
    assert len(r.retrieve_bm25_candidates("bail", candidate_k=2)) == 2 and len(bm25_sql(cur)) == 1
    assert r.bm25_stats["fills"] == 0
    print("✓ and_or fill passed")


def test_and_or_timeout():
    print("--- Testing and_or fill timeout ---")
    cur = FakeCursor(["dowri"], STATS, [row(7, 3)], [], or_error=Timeout("canceling statement due to statement timeout"))
    r, conn = make_retriever("and_or", cur)
    out = r.retrieve_bm25_candidates("dowry", candidate_k=5)
    assert [c["chunk_id"] for c in out] == [7] and conn.rolled_back == 1
    assert r.bm25_stats["fill_timeouts"] == 1
    # Any other database error is not swallowed.
    other = sys.modules["psycopg2"].Error("syntax error")
    cur = FakeCursor(["dowri"], STATS, [], [], or_error=other)
    r, _ = make_retriever("and_or", cur)
    try:
        r.retrieve_bm25_candidates("dowry", candidate_k=5)
    except sys.modules["psycopg2"].Error:
        pass
    else:
        raise AssertionError("non-timeout error swallowed")
    print("✓ timeout fallback passed")


def test_lookup_extraction():
    print("--- Testing citation / title / named-case extraction ---")
    assert lookup_strings("AIR 1978 SC 597") == ["AIR 1978 SC 597"]
    assert lookup_strings("What was held in Maneka Gandhi v. Union of India?") == ["Maneka Gandhi v Union of India"]
    assert lookup_strings("Is bail a right?") == []
    assert named_case_phrases("Explain the Maneka Gandhi case") == ["maneka gandhi"]
    assert named_case_phrases("Is the State of Punjab case relevant?") == ["state of punjab"]
    for generic in ("Explain this case", "In the bail case of the accused", "The Constitutional case law"):
        assert named_case_phrases(generic) == [], generic
    print("✓ extraction passed")


class FakeGraph:
    def __init__(self, results):
        self.results, self.calls = results, []

    def resolve_citations(self, strings):
        self.calls.append(strings)
        return [self.results.get(s, {"judgment_id": None, "method": "unresolved"}) for s in strings]


def test_lookup_and_pinning():
    print("--- Testing lookup pinning ---")
    graph = FakeGraph({
        "AIR 1978 SC 597": {"judgment_id": 11, "method": "reporter"},
        "Maneka Gandhi v Union of India": {"judgment_id": 11, "method": "exact_name"},
        "Ram v Shyam": {"judgment_id": 12, "method": "fuzzy"},  # not trusted
    })
    cur = FakeCursor([], None, [], [])
    r, _ = make_retriever("and", cur, graph)
    assert r.lookup_named_judgments("AIR 1978 SC 597 and Maneka Gandhi v. Union of India") == [(11, "reporter")]
    assert r.lookup_named_judgments("Ram v. Shyam") == []
    assert r.lookup_named_judgments("Is bail a right?") == [] and graph.calls[-1] != []

    # "<Name> case" -> party-name match, refused when it names too many judgments.
    cur = FakeCursor([], None, [], [], alias_ids=[21, 22])
    r, _ = make_retriever("and", cur, FakeGraph({}))
    assert r.lookup_named_judgments("Explain the Maneka Gandhi case") == [(21, "party_name"), (22, "party_name")]
    _, params = next((s, p) for s, p in cur.executed if "judgment_aliases" in s)
    assert params == ("maneka gandhi v %", "% v maneka gandhi", 5)
    cur = FakeCursor([], None, [], [], alias_ids=range(30, 35))
    r, _ = make_retriever("and", cur, FakeGraph({}))
    assert r.lookup_named_judgments("the State of Punjab case") == []

    # Off without a graph manager or when disabled.
    r, _ = make_retriever("and", FakeCursor([], None, [], []), None)
    assert r.lookup_named_judgments("AIR 1978 SC 597") == []
    r, _ = make_retriever("and", FakeCursor([], None, [], []), graph, case_lookup=False)
    assert r.lookup_named_judgments("AIR 1978 SC 597") == []

    # Pinned first in stage 1, ahead of fused candidates, no duplicate chunk.
    cur = FakeCursor([], None, [row(911, 11), row(5, 2)], [])
    r, _ = make_retriever("and", cur, graph)
    r.retrieve_candidate_chunks = lambda q, candidate_k, similarity_threshold: [
        {"chunk_id": 6, "judgment_id": 3, "similarity": 0.5}]
    out = r.retrieve_judgment_candidates("AIR 1978 SC 597", candidate_k=3)
    assert [c["chunk_id"] for c in out] == [911, 6, 5]  # 6 is dense rank 1, 5 BM25 rank 2
    assert out[0]["exact_match"] and out[0]["lookup_method"] == "reporter" and out[0]["rrf_score"] == 1.0
    assert sum(c["chunk_id"] == 911 for c in out) == 1
    print("✓ lookup pinning passed")


def test_rerank_keep_pinned():
    print("--- Testing rerank_keep_pinned ---")
    rr = CrossEncoderReranker.__new__(CrossEncoderReranker)

    class Model:
        def predict(self, pairs):
            return [len(t) for _, t in pairs]  # longer text scores higher

    rr.model = Model()
    chunks = [{"text": "a"}, {"text": "pinned", "exact_match": True}, {"text": "ccc"}, {"text": "bb"}]
    assert [c["text"] for c in rr.rerank_keep_pinned("q", chunks, top_n=3)] == ["pinned", "ccc", "bb"]
    assert [c["text"] for c in rr.rerank_keep_pinned("q", chunks, top_n=1)] == ["pinned"]
    assert [c["text"] for c in rr.rerank_keep_pinned("q", [{"text": "x"}], top_n=2)] == ["x"]
    print("✓ rerank_keep_pinned passed")


if __name__ == "__main__":
    test_modes_and_defaults()
    test_select_or_terms()
    test_and_mode()
    test_or_mode()
    test_and_or_fill()
    test_and_or_timeout()
    test_lookup_extraction()
    test_lookup_and_pinning()
    test_rerank_keep_pinned()
    print("\nAll BM25 / case-lookup smoke tests passed.")
