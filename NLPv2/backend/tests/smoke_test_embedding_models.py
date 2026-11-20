"""Offline checks for retrieval/embedding_models.py and the retriever's table
switch. No model is loaded (a stub stands in for sentence-transformers) and no
database is needed.

    python tests/smoke_test_embedding_models.py
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _StubST:
    def __init__(self, model_id, device=None, **kw):
        self.prompts = {"query": "Instruct: find\nQuery:"} if "prompted" in model_id else {}
        self.max_seq_length = 512
        self.seen = []

    def get_sentence_embedding_dimension(self):
        return 12

    def encode(self, texts, **kw):
        self.seen.extend(texts)
        return np.array([[len(t) + i for i in range(12)] for t in texts], dtype=np.float32)


if "sentence_transformers" not in sys.modules:
    _stub = types.ModuleType("sentence_transformers")
    _stub.SentenceTransformer = _StubST
    sys.modules["sentence_transformers"] = _stub
for _mod in ("psycopg2",):
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = types.ModuleType(_mod)

from retrieval import embedding_models as em  # noqa: E402


def _spec(**kw):
    base = dict(key="t", model_id="stub/prompted", dim=8, judgment_table="judgment_embeddings_t",
                statute_table="statute_embeddings_t")
    return em.EmbeddingSpec(**{**base, **kw})


def test_registry():
    print("--- Testing registry ---")
    old = os.environ.pop(em.ENV_VAR, None)
    try:
        spec = em.get_spec()
        assert spec.key == "bge" and spec.is_baseline
        assert (spec.judgment_table, spec.statute_table, spec.dim) == ("judgment_embeddings", "statute_embeddings", 768)
        for key in em.available():
            s = em.get_spec(key)
            if not s.is_baseline:
                assert s.judgment_table != "judgment_embeddings" and s.statute_table != "statute_embeddings", key
        try:
            em.get_spec("no-such-model")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown model accepted")
    finally:
        if old is not None:
            os.environ[em.ENV_VAR] = old
    for bad in (dict(judgment_table="x; drop table y"), dict(vector_type="bit"), dict(dim=0)):
        try:
            _spec(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad spec accepted: {bad}")
    _spec(dim=1536, vector_type="halfvec").check_indexable()
    for too_wide in (dict(dim=2048), dict(vector_type="halfvec", dim=4096)):
        try:
            _spec(**too_wide).check_indexable()  # constructible (AILA pool), not storable
        except ValueError:
            pass
        else:
            raise AssertionError(f"unindexable width accepted: {too_wide}")
    assert _spec(vector_type="halfvec", dim=2048).cosine_ops == "halfvec_cosine_ops"
    assert _spec().threshold(0.2) == 0.2 and _spec(dense_threshold=-1.0).threshold(0.2) == -1.0
    print("✓ registry passed")


def test_encode():
    print("--- Testing encode (prompts, truncation, normalisation) ---")
    spec = _spec(query_prompt=None)
    model = em.load_model(spec)
    q = em.encode(model, spec, ["abc"], "query")
    assert q.shape == (1, 8) and abs(np.linalg.norm(q[0]) - 1) < 1e-6
    assert model.seen[-1] == "Instruct: find\nQuery:abc"  # the model's own query prompt
    em.encode(model, spec, ["abc"], "document")
    assert model.seen[-1] == "abc"
    fixed = _spec(query_prompt="Q: ")
    em.encode(model, fixed, ["x"], "query")
    assert model.seen[-1] == "Q: x"
    try:
        em.load_model(_spec(dim=16))
    except ValueError:
        pass
    else:
        raise AssertionError("spec wider than the model accepted")
    assert em.vector_literal([1.0, 0.5]) == "[1.0,0.5]"
    print("✓ encode passed")


def test_retriever_tables():
    print("--- Testing retriever table switch ---")
    try:
        from retrieval.retriever import LegalRetriever
    except ImportError as e:  # graph deps missing locally
        print(f"  skipped ({e})")
        return
    r = LegalRetriever.__new__(LegalRetriever)
    assert r.embedding_spec.is_baseline
    sql = "FROM judgment_embeddings je ... statute_embeddings se ... %s::vector"
    assert r._dense_sql(sql) == sql
    r.embedding_spec = _spec(vector_type="halfvec")
    assert r._dense_sql(sql) == "FROM judgment_embeddings_t je ... statute_embeddings_t se ... %s::halfvec"
    print("✓ retriever table switch passed")


if __name__ == "__main__":
    test_registry()
    test_encode()
    test_retriever_tables()
    print("\nAll embedding-model smoke tests passed.")
