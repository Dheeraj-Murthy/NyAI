"""Embedding models the retriever can use, each with its own pgvector tables.

The BGE tables (judgment_embeddings / statute_embeddings) are the production
baseline and are never written by the experiment tooling. Every other model
stores its vectors in its own tables, keyed on the same chunk_id / section_id
and filled from the existing chunk text by ingestion/backfill_embeddings.py,
so the chunks being compared are identical and the baseline stays untouched.

RAG_EMBEDDING_MODEL picks the model retrieval reads (default "bge"). It is not
called EMBEDDING_MODEL because ingestion/.env.example already sets that name to
an unused MiniLM value, and load_dotenv() would pick it up.

This module imports no database driver, so the AILA pool benchmark can use it
without Postgres.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

ENV_VAR = "RAG_EMBEDDING_MODEL"
BASELINE = "bge"
# <repo>/models/: fine-tuned checkpoints (dvc add them; they are not in git).
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_id: str  # Hugging Face id or a local directory (e.g. a fine-tuned checkpoint)
    dim: int  # dimension stored in Postgres; the model's output is cut to it if longer
    judgment_table: str
    statute_table: str
    vector_type: str = "vector"  # "vector" (HNSW up to 2,000 dims) or "halfvec" (up to 4,000)
    # Text put before queries / documents. None = the model's own prompt from
    # its sentence-transformers config, if it has one.
    query_prompt: Optional[str] = ""
    document_prompt: Optional[str] = ""
    max_seq_length: Optional[int] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: Optional[str] = None  # e.g. "bfloat16" for large models on a GPU
    # Retrieval's dense cut-off (0.2) was tuned on BGE's similarity scale.
    # When set, this replaces it for this model; -1 keeps every top-k result.
    dense_threshold: Optional[float] = None

    def __post_init__(self):
        for name in (self.judgment_table, self.statute_table):
            if not _IDENT_RE.match(name):
                raise ValueError(f"bad table name {name!r}")
        if self.vector_type not in ("vector", "halfvec"):
            raise ValueError(f"vector_type must be vector or halfvec, got {self.vector_type!r}")
        index_limit = 2000 if self.vector_type == "vector" else 4000
        if self.dim > index_limit:
            raise ValueError(f"{self.key}: {self.dim} dims exceeds pgvector's HNSW limit for {self.vector_type}")

    @property
    def is_baseline(self) -> bool:
        return self.key == BASELINE

    @property
    def cosine_ops(self) -> str:
        return f"{self.vector_type}_cosine_ops"

    def threshold(self, caller_threshold: float) -> float:
        return caller_threshold if self.dense_threshold is None else self.dense_threshold


def _registry() -> Dict[str, EmbeddingSpec]:
    """Built on each call so env overrides (model path, dims) are read late."""
    specs = [
        EmbeddingSpec(
            key=BASELINE,
            model_id="BAAI/bge-base-en-v1.5",
            dim=768,
            judgment_table="judgment_embeddings",
            statute_table="statute_embeddings",
        ),
        # InLegalBERT (law-ai, ICAIL 2023): LEGAL-BERT further pre-trained on
        # Indian SC/HC judgments. A masked-language model, not trained to
        # embed: used as is it is mean-pooled (sentence-transformers' default
        # for a plain encoder) — the zero-shot reference point.
        EmbeddingSpec(
            key="inlegal",
            model_id="law-ai/InLegalBERT",
            dim=768,
            judgment_table="judgment_embeddings_inlegal",
            statute_table="statute_embeddings_inlegal",
            max_seq_length=512,
            dense_threshold=-1.0,  # its cosine scale isn't BGE's; keep the top k
        ),
        # InLegalBERT fine-tuned on citation pairs (training/train_embedder.py).
        EmbeddingSpec(
            key="inlegal_ft",
            model_id=os.environ.get("INLEGAL_FT_PATH", os.path.join(MODELS_DIR, "inlegal-ft")),
            dim=768,
            judgment_table="judgment_embeddings_inlegal_ft",
            statute_table="statute_embeddings_inlegal_ft",
            max_seq_length=512,
            dense_threshold=-1.0,
        ),
        # Control: BGE fine-tuned on the same pairs. Separates "Indian legal
        # pre-training helps" from "fine-tuning on our citations helps".
        EmbeddingSpec(
            key="bge_ft",
            model_id=os.environ.get("BGE_FT_PATH", os.path.join(MODELS_DIR, "bge-ft")),
            dim=768,
            judgment_table="judgment_embeddings_bge_ft",
            statute_table="statute_embeddings_bge_ft",
            max_seq_length=512,
            dense_threshold=-1.0,
        ),
    ]
    return {s.key: s for s in specs}


def available() -> List[str]:
    return sorted(_registry())


def get_spec(key: Optional[str] = None) -> EmbeddingSpec:
    key = key or os.environ.get(ENV_VAR) or BASELINE
    reg = _registry()
    if key not in reg:
        raise ValueError(f"unknown embedding model {key!r} ({ENV_VAR}); choose from {sorted(reg)}")
    return reg[key]


def load_model(spec: EmbeddingSpec, device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    model_kwargs = {}
    if spec.dtype:
        import torch
        import transformers

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        # transformers renamed torch_dtype -> dtype in 4.56
        model_kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = getattr(torch, spec.dtype)
    kwargs = {"trust_remote_code": spec.trust_remote_code}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    model = SentenceTransformer(spec.model_id, device=device, **kwargs)
    if spec.max_seq_length:
        model.max_seq_length = spec.max_seq_length
    out_dim = model.get_sentence_embedding_dimension()
    if out_dim is not None and out_dim < spec.dim:
        raise ValueError(f"{spec.model_id} outputs {out_dim} dims, spec stores {spec.dim}")
    return model


def prompt_for(model, spec: EmbeddingSpec, kind: str) -> str:
    """The text put before a query (kind="query") or document (kind="document")."""
    p = spec.query_prompt if kind == "query" else spec.document_prompt
    if p is None:
        p = (getattr(model, "prompts", None) or {}).get(kind, "")
    return p


def encode_full(model, spec: EmbeddingSpec, texts: Sequence[str], kind: str,
                batch_size: int = 32, show_progress_bar: bool = False):
    """The model's raw float32 output at its full width (prompt applied)."""
    import numpy as np

    prefix = prompt_for(model, spec, kind)
    return model.encode(
        [prefix + (t or "") for t in texts], batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=show_progress_bar,
    ).astype(np.float32)


def truncate(emb, dim: int):
    """First dim dimensions, re-normalised to unit length (Matryoshka truncation)."""
    import numpy as np

    emb = emb[:, :dim]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1.0, norms)


def encode(model, spec: EmbeddingSpec, texts: Sequence[str], kind: str,
           batch_size: int = 32, show_progress_bar: bool = False):
    """Unit-length float32 vectors of spec.dim (cut, then re-normalised, when
    the model outputs more dimensions)."""
    return truncate(encode_full(model, spec, texts, kind, batch_size, show_progress_bar), spec.dim)


def model_fingerprint(spec: EmbeddingSpec) -> str:
    """Identifies the exact weights a table was filled with. A local directory
    (a fine-tuned checkpoint) is fingerprinted by its files' names, sizes and
    mtimes, so retraining into the same path is noticed."""
    import hashlib

    if os.path.isdir(spec.model_id):
        h = hashlib.sha256()
        for root, _, files in sorted(os.walk(spec.model_id)):
            for name in sorted(files):
                st = os.stat(os.path.join(root, name))
                h.update(f"{os.path.relpath(os.path.join(root, name), spec.model_id)}:{st.st_size}:{int(st.st_mtime)}\n".encode())
        return f"dir:{h.hexdigest()[:16]}"
    return f"hf:{spec.model_id}@{spec.revision or 'main'}"


def vector_literal(values: Sequence[float]) -> str:
    """pgvector text form, accepted by both ::vector and ::halfvec."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
