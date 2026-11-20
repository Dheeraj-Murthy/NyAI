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
from typing import Any, Dict, List, Optional, Sequence

ENV_VAR = "RAG_EMBEDDING_MODEL"
BASELINE = "bge"
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
        if self.dim < 1:
            raise ValueError(f"{self.key}: dim must be positive")

    def check_indexable(self) -> None:
        """pgvector's HNSW index takes up to 2,000 dims as vector, 4,000 as
        halfvec. Checked when storing, not here: the AILA pool can score any width."""
        limit = 2000 if self.vector_type == "vector" else 4000
        if self.dim > limit:
            raise ValueError(f"{self.key}: {self.dim} dims exceeds pgvector's HNSW limit for {self.vector_type} ({limit})")

    @property
    def is_baseline(self) -> bool:
        return self.key == BASELINE

    @property
    def cosine_ops(self) -> str:
        return f"{self.vector_type}_cosine_ops"

    def threshold(self, caller_threshold: float) -> float:
        return caller_threshold if self.dense_threshold is None else self.dense_threshold


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer")


def _instruct_prompt(env_name: str) -> Optional[str]:
    """Qwen3-Embedding's query format with a task of our own (env), or None
    for the model's own query prompt from its sentence-transformers config."""
    task = os.environ.get(env_name)
    return f"Instruct: {task}\nQuery:" if task else None


QWEN_KEYS = ("euler", "qwen3_8b", "qwen3_06b")


def _qwen_specs() -> List[EmbeddingSpec]:
    """Qwen3-Embedding family. Built apart from the rest so a bad env value
    here (EULER_DIM=abc) fails only these models, never the baseline."""
    # The 8B models output 4,096 dims; HNSW indexes at most 4,000 (halfvec),
    # so they are stored Matryoshka-truncated. EULER_DIM sets the width for
    # both 8B models (tables are named by it, so widths never share a table).
    dim_8b = _env_int("EULER_DIM", 2048)
    vtype_8b = "halfvec" if dim_8b > 2000 or os.environ.get("EULER_HALFVEC", "1") == "1" else "vector"
    big = dict(max_seq_length=1536, dtype="bfloat16", dense_threshold=-1.0, query_prompt=None, document_prompt=None)
    return [
        # Euler-Legal-Embedding-V1: a LoRA legal fine-tune of
        # Qwen3-Embedding-8B (last-token pooling, 1536 tokens). Unverified
        # third-party upload: pin EULER_REVISION to a commit after reviewing
        # the repo. trust_remote_code stays off unless EULER_TRUST_REMOTE_CODE=1
        # (Qwen3 loads natively in recent transformers).
        EmbeddingSpec(
            key="euler",
            model_id=os.environ.get("EULER_MODEL", "Mira190/Euler-Legal-Embedding-V1"),
            revision=os.environ.get("EULER_REVISION") or None,
            trust_remote_code=os.environ.get("EULER_TRUST_REMOTE_CODE") == "1",
            dim=dim_8b,
            vector_type=vtype_8b,
            judgment_table=f"judgment_embeddings_euler_{dim_8b}",
            statute_table=f"statute_embeddings_euler_{dim_8b}",
            **{**big, "query_prompt": _instruct_prompt("QWEN_QUERY_INSTRUCT")},
        ),
        # Control: the base model Euler was fine-tuned from. Euler minus this
        # is what the legal fine-tune adds.
        EmbeddingSpec(
            key="qwen3_8b",
            model_id="Qwen/Qwen3-Embedding-8B",
            revision=os.environ.get("QWEN3_8B_REVISION") or None,
            dim=dim_8b,
            vector_type=vtype_8b,
            judgment_table=f"judgment_embeddings_qwen3_8b_{dim_8b}",
            statute_table=f"statute_embeddings_qwen3_8b_{dim_8b}",
            **{**big, "query_prompt": _instruct_prompt("QWEN_QUERY_INSTRUCT")},
        ),
        # Light option: same family, 0.6B, 1,024 dims (fits vector + HNSW as is).
        EmbeddingSpec(
            key="qwen3_06b",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            revision=os.environ.get("QWEN3_06B_REVISION") or None,
            dim=_env_int("QWEN3_06B_DIM", 1024),
            judgment_table=f"judgment_embeddings_qwen3_06b_{_env_int('QWEN3_06B_DIM', 1024)}",
            statute_table=f"statute_embeddings_qwen3_06b_{_env_int('QWEN3_06B_DIM', 1024)}",
            **{**big, "query_prompt": _instruct_prompt("QWEN_QUERY_INSTRUCT")},
        ),
    ]


def _registry() -> Dict[str, Any]:
    """key -> spec, or the ValueError that building it raised. Built on each
    call so env overrides (model path, dims) are read late."""
    specs: Dict[str, Any] = {
        BASELINE: EmbeddingSpec(
            key=BASELINE,
            model_id="BAAI/bge-base-en-v1.5",
            dim=768,
            judgment_table="judgment_embeddings",
            statute_table="statute_embeddings",
        ),
    }
    try:
        specs.update({s.key: s for s in _qwen_specs()})
    except ValueError as e:
        specs.update({k: e for k in QWEN_KEYS})
    return specs


def available() -> List[str]:
    return sorted(_registry())


def get_spec(key: Optional[str] = None) -> EmbeddingSpec:
    key = key or os.environ.get(ENV_VAR) or BASELINE
    reg = _registry()
    if key not in reg:
        raise ValueError(f"unknown embedding model {key!r} ({ENV_VAR}); choose from {sorted(reg)}")
    if isinstance(reg[key], ValueError):
        raise ValueError(f"embedding model {key!r} is misconfigured: {reg[key]}")
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
    (a fine-tuned checkpoint) is fingerprinted by its files' names and
    contents, so retraining into the same path is noticed while a copy or a
    `dvc pull` of the same weights is not (seconds for a BERT-size model)."""
    import hashlib

    if os.path.isdir(spec.model_id):
        h = hashlib.sha256()
        for root, dirs, files in os.walk(spec.model_id):
            dirs.sort()
            for name in sorted(files):
                path = os.path.join(root, name)
                h.update(os.path.relpath(path, spec.model_id).encode() + b"\0")
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
        return f"dir:{h.hexdigest()[:16]}"
    return f"hf:{spec.model_id}@{spec.revision or 'main'}"


def vector_literal(values: Sequence[float]) -> str:
    """pgvector text form, accepted by both ::vector and ::halfvec."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
