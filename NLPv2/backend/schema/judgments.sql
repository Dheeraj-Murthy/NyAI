-- Legal RAG schema, part 1: judgments, citation graph and chat history.
-- Statutes are in statutes.sql.
--
-- Additive and idempotent: safe on an empty database, on one that already
-- has data, and to run more than once. Columns added after a table first
-- shipped are created with the table AND added by ALTER ... IF NOT EXISTS,
-- so older databases are brought up to date without losing anything.
--
-- Applied together with the other file in this folder by
-- deploy/db_setup/init_db.sh and backend/db_schema.py; no code path keeps
-- its own copy of any table.

CREATE EXTENSION IF NOT EXISTS vector;

-- Fingerprint (sha256 of judgments.sql + statutes.sql) of the schema last
-- applied. db_schema.py skips re-applying when it matches, so apps starting
-- up don't take table locks while ingestion is writing.
CREATE TABLE IF NOT EXISTS schema_version (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),   -- single row
    sha256 TEXT NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ===========================================================================
-- Judgments
-- ===========================================================================

CREATE TABLE IF NOT EXISTS judgments (
    id SERIAL PRIMARY KEY,
    petitioner TEXT,
    respondent TEXT,
    court TEXT,
    date_of_judgment DATE,
    bench TEXT[],
    citations JSONB,
    judgment_text TEXT,
    -- Original PDF, relative to the repository root (see backend/documents.py).
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE judgments ADD COLUMN IF NOT EXISTS source_file TEXT;
-- eCourts case reference number and disposal outcome (set by ingestion/ingest_aws.py).
ALTER TABLE judgments ADD COLUMN IF NOT EXISTS cnr TEXT;
ALTER TABLE judgments ADD COLUMN IF NOT EXISTS disposal_nature TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS judgments_cnr_uidx ON judgments(cnr) WHERE cnr IS NOT NULL;

CREATE TABLE IF NOT EXISTS judgment_chunks (
    chunk_id SERIAL PRIMARY KEY,
    judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
    section TEXT CHECK (section IN ('facts', 'issues', 'arguments', 'ratio', 'judgment')),
    content TEXT,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 768-dim embeddings (BGE-base-en-v1.5)
CREATE TABLE IF NOT EXISTS judgment_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES judgment_chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS judgment_embedding_hnsw_idx
    ON judgment_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS judgment_chunks_judgment_id_idx ON judgment_chunks(judgment_id);
CREATE INDEX IF NOT EXISTS judgment_chunks_section_idx ON judgment_chunks(section);
-- BM25-style full-text search (hybrid retrieval, alongside pgvector)
CREATE INDEX IF NOT EXISTS judgment_chunks_content_tsv_idx ON judgment_chunks USING GIN (content_tsv);


-- ===========================================================================
-- Citation graph
-- ===========================================================================

CREATE TABLE IF NOT EXISTS citation_edges (
    edge_id SERIAL PRIMARY KEY,
    source_judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
    target_judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,  -- NULL = not matched
    cited_text TEXT NOT NULL,
    relationship_type VARCHAR(50) DEFAULT 'cited',
    -- How the target was found: reporter | exact_name | fuzzy | ambiguous | unresolved.
    -- NULL on edges built before the resolver existed.
    resolution_method VARCHAR(20),
    resolution_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE citation_edges ADD COLUMN IF NOT EXISTS resolution_method VARCHAR(20);
ALTER TABLE citation_edges ADD COLUMN IF NOT EXISTS resolution_score REAL;

CREATE INDEX IF NOT EXISTS idx_citation_edges_source ON citation_edges(source_judgment_id);
CREATE INDEX IF NOT EXISTS idx_citation_edges_target ON citation_edges(target_judgment_id);
-- Prevent duplicate edges on re-runs; NULL targets fold to a sentinel so
-- unresolved citations are deduplicated too.
CREATE UNIQUE INDEX IF NOT EXISTS idx_citation_edges_unique
    ON citation_edges(source_judgment_id, COALESCE(target_judgment_id, -1), cited_text);

-- Every normalised name and reporter citation a judgment can be found by.
-- kind = 'case_name' ("state of bombay v f n balsara") or
--        'reporter'  ("air|1953|sc|75", "scr|1955|1|777", "scr|1955||777").
-- One alias may map to several judgments; that is how ambiguity is detected.
CREATE TABLE IF NOT EXISTS judgment_aliases (
    alias_norm TEXT NOT NULL,
    judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
    kind VARCHAR(20) NOT NULL,
    PRIMARY KEY (alias_norm, judgment_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_judgment_aliases_judgment ON judgment_aliases(judgment_id);

-- Precomputed by compute_graph_stats.py so the API never runs PageRank while serving.
CREATE TABLE IF NOT EXISTS judgment_graph_stats (
    judgment_id INTEGER PRIMARY KEY REFERENCES judgments(id) ON DELETE CASCADE,
    in_degree INTEGER NOT NULL DEFAULT 0,
    out_degree INTEGER NOT NULL DEFAULT 0,
    pagerank DOUBLE PRECISION NOT NULL DEFAULT 0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_judgment_graph_stats_pagerank
    ON judgment_graph_stats(pagerank DESC);

-- Optional: pg_trgm makes /graph/search fast. If the role can't create the
-- extension, search falls back to a slower scan (graph/pg_store.py checks)
-- instead of the whole schema failing.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_trgm unavailable (%); case search will use a slower scan.', SQLERRM;
END
$$;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        CREATE INDEX IF NOT EXISTS idx_judgment_aliases_trgm
            ON judgment_aliases USING GIN (alias_norm gin_trgm_ops);
    END IF;
END
$$;


-- ===========================================================================
-- Chat history (backend/chat_store.py)
-- ===========================================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Shown in the chat list; set from the first question unless renamed.
    title TEXT,
    -- Running summary of the oldest messages, made when the conversation
    -- outgrew a model's context window; covers messages up to summary_upto.
    summary TEXT,
    summary_upto INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,          -- what the user sees
    prompt_text TEXT,               -- what the model sees as history (answer without the source list)
    retrieval_query TEXT,           -- user turns: the query actually used for retrieval
    -- Assistant turns: citations, sources, confidence, model, and the
    -- citation/source refs (judgment ID / statute section ID) that make
    -- citations clickable.
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, message_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_active ON chat_sessions(last_active DESC);
