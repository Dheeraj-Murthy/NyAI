"""
Postgres-backed citation graph queries.

Walks citation_edges with indexed queries instead of holding the whole graph
in memory, and reads precomputed PageRank from judgment_graph_stats. Every
method degrades instead of failing when the newer tables are missing:

  - no judgment_graph_stats rows -> PageRank computed once here from edge IDs
                                    and cached (what the API did before)
  - no judgment_aliases table    -> search/resolve match on judgments directly
  - no pg_trgm                   -> search uses LIKE without the trigram index

Edges are de-duplicated per (source, target): several rows for the same pair
(different cited_text) count once, keeping the strongest relationship.
Self-citations are ignored.
"""

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from psycopg2.pool import ThreadedConnectionPool

from graph.citation_resolver import (
    CitationResolver,
    _year_of,
    case_name_alias,
    normalize_case_name,
    parse_reporter_citations,
)

EDGE_FILTER = "target_judgment_id IS NOT NULL AND source_judgment_id <> target_judgment_id"

# Lower = stronger; matches CitationGraphManager.get_precedent_summary's ordering.
REL_RANK_SQL = (
    "CASE COALESCE(relationship_type, 'cited') WHEN 'overruled' THEN 0 WHEN 'followed' THEN 1 "
    "WHEN 'distinguished' THEN 2 WHEN 'referred' THEN 3 WHEN 'cited' THEN 4 ELSE 5 END"
)

# How long table-availability checks and fallback caches are trusted, so the
# API notices a migration or a stats run without a restart.
CHECK_TTL_SECONDS = 300

# Orders landmark_cases() can rank by.
LANDMARK_RANKINGS = ("pagerank", "citations_per_year", "in_degree")

# Citations per year = times cited / years since the judgment, counted up to
# the newest judgment in the corpus rather than today, so the corpus cut-off
# doesn't penalise recent cases. A case counts as at least this many years
# old, so a couple of citations in its first year don't outrank decades of
# steady citation.
MIN_CITATION_YEARS = 5


def citations_per_year(in_degree: int, judgment_year: Optional[int], end_year: Optional[int]) -> float:
    """Times cited per year since the judgment; 0 when the judgment is undated."""
    if not judgment_year or not end_year:
        return 0.0
    return in_degree / max(MIN_CITATION_YEARS, end_year - judgment_year + 1)


def rank_landmarks(rows: List[Tuple[int, float, int, float]], rank_by: str, limit: int) -> List[Tuple[int, float, int, float]]:
    """Top `limit` of (judgment_id, pagerank, in_degree, citations_per_year)
    rows in the given order — the in-Python twin of landmark_cases()' SQL."""
    keys = {
        "pagerank": lambda r: (round(r[1], 6), r[2]),
        "citations_per_year": lambda r: (r[3], r[2]),
        "in_degree": lambda r: (r[2], r[1]),
    }
    return sorted(rows, key=keys[rank_by], reverse=True)[:limit]


def make_label(petitioner: Optional[str], respondent: Optional[str]) -> str:
    return f"{petitioner or 'Unknown'} v. {respondent or 'Unknown'}"


def _like(text: str) -> str:
    """An ILIKE pattern matching text anywhere, with its wildcards escaped."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class PostgresGraphStore:
    def __init__(self, conn_kwargs: Dict[str, Any], max_connections: int = 5):
        self._conn_kwargs = conn_kwargs
        self._max_connections = max_connections
        self._pool: Optional[ThreadedConnectionPool] = None
        self._lock = threading.Lock()
        self._checks: Dict[str, Tuple[float, Any]] = {}

    # -- connections ----------------------------------------------------------

    @contextmanager
    def _cursor(self):
        with self._lock:
            if self._pool is None:
                self._pool = ThreadedConnectionPool(1, self._max_connections, **self._conn_kwargs)
        conn = self._pool.getconn()
        broken = False
        try:
            conn.autocommit = True  # read-only queries; no transaction left open
            with conn.cursor() as cur:
                yield cur
        except Exception:
            broken = conn.closed != 0
            raise
        finally:
            self._pool.putconn(conn, close=broken)

    def _cached(self, name: str, compute):
        """compute() result cached for CHECK_TTL_SECONDS."""
        now = time.monotonic()
        hit = self._checks.get(name)
        if hit and now - hit[0] < CHECK_TTL_SECONDS:
            return hit[1]
        value = compute()
        self._checks[name] = (now, value)
        return value

    def _table_exists(self, table: str) -> bool:
        def check():
            with self._cursor() as cur:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{table}",))
                return bool(cur.fetchone()[0])
        return self._cached(f"table:{table}", check)

    def _stats_ready(self) -> bool:
        def check():
            if not self._table_exists("judgment_graph_stats"):
                return False
            with self._cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM judgment_graph_stats)")
                return bool(cur.fetchone()[0])
        return self._cached("stats_ready", check)

    def _trigram_ready(self) -> bool:
        def check():
            with self._cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
                return bool(cur.fetchone()[0])
        return self._cached("trigram", check)

    # -- basics ---------------------------------------------------------------

    def edge_count(self) -> int:
        if not self._table_exists("citation_edges"):
            return 0
        with self._cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT source_judgment_id, target_judgment_id "
                f"FROM citation_edges WHERE {EDGE_FILTER}) e"
            )
            return int(cur.fetchone()[0])

    def _judgments(self, cur, ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        cur.execute(
            "SELECT id, petitioner, respondent, court, date_of_judgment FROM judgments WHERE id = ANY(%s)",
            (list(ids),),
        )
        return {
            row[0]: {
                "label": make_label(row[1], row[2]),
                "court": row[3],
                "date": str(row[4]),
            }
            for row in cur.fetchall()
        }

    def judgment_info(self, judgment_id: int) -> Dict[str, Any]:
        """Label, court and date of one judgment."""
        with self._cursor() as cur:
            info = self._judgments(cur, [judgment_id])
        return info.get(judgment_id) or {"label": f"Case #{judgment_id}", "court": None, "date": None}

    def _exists(self, cur, judgment_id: int) -> bool:
        cur.execute("SELECT EXISTS (SELECT 1 FROM judgments WHERE id = %s)", (judgment_id,))
        return bool(cur.fetchone()[0])

    # -- PageRank / degrees -----------------------------------------------------

    def _fallback_stats(self) -> Dict[int, Tuple[float, int]]:
        """{judgment_id: (pagerank, in_degree)} computed here from edge IDs,
        for databases where compute_graph_stats.py hasn't run yet."""
        def compute():
            with self._cursor() as cur:
                cur.execute("SELECT id FROM judgments")
                nodes = [r[0] for r in cur.fetchall()]
                cur.execute(
                    f"SELECT DISTINCT source_judgment_id, target_judgment_id FROM citation_edges "
                    f"WHERE {EDGE_FILTER}"
                )
                edges = cur.fetchall()
            g = nx.DiGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            pr = nx.pagerank(g) if edges else {}
            return {n: (pr.get(n, 0.0), g.in_degree(n)) for n in g.nodes}
        return self._cached("fallback_stats", compute)

    def scores_for(self, ids: List[int]) -> Dict[int, float]:
        if not ids:
            return {}
        if self._stats_ready():
            with self._cursor() as cur:
                cur.execute(
                    "SELECT judgment_id, pagerank FROM judgment_graph_stats WHERE judgment_id = ANY(%s)",
                    (list(ids),),
                )
                return {r[0]: float(r[1]) for r in cur.fetchall()}
        stats = self._fallback_stats()
        return {i: stats[i][0] for i in ids if i in stats}

    def all_scores(self) -> Dict[int, float]:
        if self._stats_ready():
            with self._cursor() as cur:
                cur.execute(
                    "SELECT judgment_id, pagerank FROM judgment_graph_stats WHERE pagerank > 0"
                )
                return {r[0]: float(r[1]) for r in cur.fetchall()}
        return {n: pr for n, (pr, _) in self._fallback_stats().items() if pr > 0}

    def _judgment_years(self) -> Tuple[Dict[int, int], Optional[int]]:
        """({judgment_id: year}, newest year in the corpus), for databases
        where compute_graph_stats.py hasn't run yet."""
        def compute():
            with self._cursor() as cur:
                cur.execute(
                    "SELECT id, EXTRACT(YEAR FROM date_of_judgment)::int FROM judgments "
                    "WHERE date_of_judgment IS NOT NULL"
                )
                years = dict(cur.fetchall())
            return years, max(years.values(), default=None)
        return self._cached("judgment_years", compute)

    def landmark_cases(self, limit: int = 10, rank_by: str = "pagerank") -> List[Dict[str, Any]]:
        """Top cases by PageRank, citations per year or total citations
        (in-degree); each ties broken by the other measures."""
        if rank_by not in LANDMARK_RANKINGS:
            raise ValueError(f"rank_by must be one of {LANDMARK_RANKINGS}")
        order = {
            "pagerank": "s.pagerank DESC, s.in_degree DESC",
            "citations_per_year": "citations_per_year DESC, s.in_degree DESC",
            "in_degree": "s.in_degree DESC, s.pagerank DESC",
        }[rank_by]
        with self._cursor() as cur:
            if self._stats_ready():
                cur.execute(
                    "WITH span AS (SELECT EXTRACT(YEAR FROM MAX(date_of_judgment))::int AS end_year "
                    "              FROM judgments) "
                    "SELECT s.judgment_id, s.pagerank, s.in_degree, "
                    # GREATEST skips NULLs, so an undated judgment needs its own branch.
                    "       CASE WHEN j.date_of_judgment IS NULL THEN 0 "
                    "            ELSE s.in_degree::float / GREATEST(%s, span.end_year "
                    "                 - EXTRACT(YEAR FROM j.date_of_judgment)::int + 1) "
                    "       END AS citations_per_year "
                    "FROM judgment_graph_stats s JOIN judgments j ON j.id = s.judgment_id CROSS JOIN span "
                    f"ORDER BY {order}, s.judgment_id LIMIT %s",
                    (MIN_CITATION_YEARS, limit),
                )
                top = [(r[0], float(r[1]), int(r[2]), float(r[3])) for r in cur.fetchall()]
            else:
                years, end_year = self._judgment_years()
                rows = [
                    (n, pr, deg, citations_per_year(deg, years.get(n), end_year))
                    for n, (pr, deg) in self._fallback_stats().items()
                ]
                top = rank_landmarks(rows, rank_by, limit)
            info = self._judgments(cur, [t[0] for t in top])
        return [
            {
                "judgment_id": n,
                "label": info.get(n, {}).get("label", f"Case #{n}"),
                "court": info.get(n, {}).get("court"),
                "date": info.get(n, {}).get("date"),
                "in_degree": deg,
                "citations_per_year": round(cpy, 3),
                "pagerank_score": round(pr, 6),
            }
            for n, pr, deg, cpy in top
        ]

    # -- traversal ------------------------------------------------------------

    def subgraph(self, judgment_id: int, depth: int = 2, max_nodes: int = 100) -> Dict[str, Any]:
        """N-hop neighbourhood, citations followed in both directions. One
        query per hop, integer IDs only while expanding; labels are fetched
        just for the nodes kept. Over max_nodes, closer nodes win, then
        higher PageRank — every hop level is kept whole before the next is
        touched, so the result stays connected. That also means once a hop
        level overshoots max_nodes nothing further out can be kept, so the
        search stops there and total_nodes becomes a lower bound
        (total_exact False) instead of expanding a landmark case's
        neighbourhood by thousands of nodes only to drop them."""
        empty = {
            "nodes": [], "edges": [], "center_id": judgment_id, "total_nodes": 0, "total_exact": True, "truncated": False,
        }
        with self._cursor() as cur:
            if not self._exists(cur, judgment_id):
                return empty

            dist = {judgment_id: 0}
            frontier = [judgment_id]
            total_exact = True
            for hop in range(1, depth + 1):
                if not frontier:
                    break
                if len(dist) > max_nodes:
                    total_exact = False
                    break
                cur.execute(
                    f"SELECT source_judgment_id, target_judgment_id FROM citation_edges "
                    f"WHERE {EDGE_FILTER} AND (source_judgment_id = ANY(%s) OR target_judgment_id = ANY(%s))",
                    (frontier, frontier),
                )
                nxt = []
                for src, tgt in cur.fetchall():
                    for n in (src, tgt):
                        if n not in dist:
                            dist[n] = hop
                            nxt.append(n)
                frontier = nxt

        total_nodes = len(dist)
        truncated = total_nodes > max_nodes
        if truncated:
            scores = self.scores_for(list(dist))
            keep = sorted(dist, key=lambda n: (dist[n], -scores.get(n, 0.0), n))[:max_nodes]
        else:
            keep = list(dist)

        with self._cursor() as cur:
            info = self._judgments(cur, keep)
            cur.execute(
                f"SELECT DISTINCT ON (source_judgment_id, target_judgment_id) "
                f"source_judgment_id, target_judgment_id, cited_text, COALESCE(relationship_type, 'cited') "
                f"FROM citation_edges WHERE {EDGE_FILTER} "
                f"AND source_judgment_id = ANY(%s) AND target_judgment_id = ANY(%s) "
                f"ORDER BY source_judgment_id, target_judgment_id, {REL_RANK_SQL}, edge_id",
                (keep, keep),
            )
            edge_rows = cur.fetchall()

        nodes = [
            {
                "id": n,
                "label": info.get(n, {}).get("label", f"Case #{n}"),
                "court": info.get(n, {}).get("court"),
                "date": info.get(n, {}).get("date"),
                "is_center": n == judgment_id,
                "hop": dist[n],
            }
            for n in keep
        ]
        edges = [
            {"source": s, "target": t, "cited_text": text or "", "relationship": rel}
            for s, t, text, rel in edge_rows
        ]
        return {
            "center_id": judgment_id,
            "nodes": nodes,
            "edges": edges,
            "total_nodes": total_nodes,
            "total_exact": total_exact,
            "truncated": truncated,
        }

    def shortest_path(self, source_id: int, target_id: int, max_hops: int = 12) -> Optional[List[int]]:
        """Shortest chain source -> ... -> target following citation
        direction. Bidirectional BFS, expanding the smaller frontier, one
        query per level.

        Pruned by date: a judgment only cites earlier ones, so every case on
        a chain falls between the target's date and the source's. Cases
        outside that window are never expanded, and a source older than its
        target has no chain at all. Cases without a date are kept."""
        with self._cursor() as cur:
            cur.execute("SELECT id, date_of_judgment FROM judgments WHERE id = ANY(%s)", ([source_id, target_id],))
            dates = dict(cur.fetchall())
            if len(dates) < len({source_id, target_id}):
                return None
            if source_id == target_id:
                return [source_id]

            newest, oldest = dates[source_id], dates[target_id]
            if newest is not None and oldest is not None and newest < oldest:
                return None
            # Applied to the case a step reaches (j); %s pairs are filled per query.
            window, window_params = "", []
            if oldest is not None:
                window += " AND (j.date_of_judgment IS NULL OR j.date_of_judgment >= %s)"
                window_params.append(oldest)
            if newest is not None:
                window += " AND (j.date_of_judgment IS NULL OR j.date_of_judgment <= %s)"
                window_params.append(newest)

            fwd: Dict[int, Optional[int]] = {source_id: None}
            bwd: Dict[int, Optional[int]] = {target_id: None}
            fwd_frontier, bwd_frontier = [source_id], [target_id]

            for _ in range(max_hops):
                if not fwd_frontier or not bwd_frontier:
                    return None
                meet: List[int] = []
                if len(fwd_frontier) <= len(bwd_frontier):
                    cur.execute(
                        f"SELECT source_judgment_id, target_judgment_id FROM citation_edges "
                        f"JOIN judgments j ON j.id = target_judgment_id "
                        f"WHERE {EDGE_FILTER} AND source_judgment_id = ANY(%s){window} "
                        f"ORDER BY source_judgment_id, target_judgment_id",
                        [fwd_frontier] + window_params,
                    )
                    nxt = []
                    for src, tgt in cur.fetchall():
                        if tgt not in fwd:
                            fwd[tgt] = src
                            nxt.append(tgt)
                            if tgt in bwd:
                                meet.append(tgt)
                    fwd_frontier = nxt
                else:
                    cur.execute(
                        f"SELECT source_judgment_id, target_judgment_id FROM citation_edges "
                        f"JOIN judgments j ON j.id = source_judgment_id "
                        f"WHERE {EDGE_FILTER} AND target_judgment_id = ANY(%s){window} "
                        f"ORDER BY target_judgment_id, source_judgment_id",
                        [bwd_frontier] + window_params,
                    )
                    nxt = []
                    for src, tgt in cur.fetchall():
                        if src not in bwd:
                            bwd[src] = tgt
                            nxt.append(src)
                            if src in fwd:
                                meet.append(src)
                    bwd_frontier = nxt
                if meet:
                    paths = [self._join_path(fwd, bwd, m) for m in meet]
                    return min(paths, key=len)
        return None

    @staticmethod
    def _join_path(fwd: Dict[int, Optional[int]], bwd: Dict[int, Optional[int]], meet: int) -> List[int]:
        head: List[int] = []
        node: Optional[int] = meet
        while node is not None:
            head.append(node)
            node = fwd[node]
        head.reverse()
        node = bwd[meet]
        while node is not None:
            head.append(node)
            node = bwd[node]
        return head

    def neighbors(
        self,
        judgment_id: int,
        direction: str = "cites",
        relationship: Optional[str] = None,
        limit: int = 25,
        offset: int = 0,
        q: Optional[str] = None,
        court: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """One page of the cases a judgment cites (direction="cites") or that
        cite it ("cited_by"), most important first. q matches the case name
        or the citation text, court the court name (both case-insensitive
        substrings); total counts the matches. None if the judgment doesn't
        exist."""
        mine, other = (
            ("source_judgment_id", "target_judgment_id")
            if direction == "cites"
            else ("target_judgment_id", "source_judgment_id")
        )
        rel_filter = "AND COALESCE(relationship_type, 'cited') = %s" if relationship else ""
        stats = self._stats_ready()
        order = "COALESCE(s.pagerank, 0) DESC, n.rank, n.id" if stats else "n.rank, n.id"
        stats_join = "LEFT JOIN judgment_graph_stats s ON s.judgment_id = n.id" if stats else ""
        where, where_params = [], []
        if q:
            # Same text as make_label, so "X v. Y" searches match what the table shows.
            where.append(
                "((COALESCE(j.petitioner, 'Unknown') || ' v. ' || COALESCE(j.respondent, 'Unknown')) ILIKE %s"
                " OR n.cited_text ILIKE %s)"
            )
            where_params += [_like(q)] * 2
        if court:
            where.append("j.court ILIKE %s")
            where_params.append(_like(court))
        outer_filter = ("WHERE " + " AND ".join(where)) if where else ""
        params: List[Any] = (
            [judgment_id] + ([relationship] if relationship else []) + where_params + [limit, offset]
        )

        with self._cursor() as cur:
            if not self._exists(cur, judgment_id):
                return None
            cur.execute(
                f"""
                SELECT n.id, j.petitioner, j.respondent, j.court, j.date_of_judgment,
                       n.rel, n.cited_text, {"s.pagerank" if stats else "NULL"}, COUNT(*) OVER ()
                FROM (
                    SELECT DISTINCT ON ({other}) {other} AS id,
                           COALESCE(relationship_type, 'cited') AS rel, cited_text,
                           {REL_RANK_SQL} AS rank
                    FROM citation_edges
                    WHERE {EDGE_FILTER} AND {mine} = %s {rel_filter}
                    ORDER BY {other}, {REL_RANK_SQL}, edge_id
                ) n
                JOIN judgments j ON j.id = n.id
                {stats_join}
                {outer_filter}
                ORDER BY {order}
                LIMIT %s OFFSET %s
                """,
                params,
            )
            rows = cur.fetchall()

        if not stats and rows:
            scores = self.scores_for([r[0] for r in rows])
            rows = [r[:7] + (scores.get(r[0]),) + r[8:] for r in rows]

        return {
            "judgment_id": judgment_id,
            "direction": direction,
            "total": int(rows[0][8]) if rows else 0,
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "judgment_id": r[0],
                    "label": make_label(r[1], r[2]),
                    "court": r[3],
                    "date": str(r[4]),
                    "relationship": r[5],
                    "cited_text": r[6] or "",
                    "pagerank_score": round(float(r[7]), 6) if r[7] is not None else None,
                }
                for r in rows
            ],
        }

    def precedent_summary(self, judgment_id: int, top_n: int = 3) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT petitioner, respondent FROM judgments WHERE id = %s", (judgment_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            label = make_label(row[0], row[1])

            def side(mine: str, other: str) -> Tuple[List[Dict[str, Any]], int]:
                cur.execute(
                    f"""
                    SELECT n.id, j.petitioner, j.respondent, n.rel, COUNT(*) OVER ()
                    FROM (
                        SELECT DISTINCT ON ({other}) {other} AS id,
                               COALESCE(relationship_type, 'cited') AS rel, {REL_RANK_SQL} AS rank
                        FROM citation_edges
                        WHERE {EDGE_FILTER} AND {mine} = %s
                        ORDER BY {other}, {REL_RANK_SQL}, edge_id
                    ) n
                    JOIN judgments j ON j.id = n.id
                    ORDER BY n.rank, n.id
                    LIMIT %s
                    """,
                    (judgment_id, top_n),
                )
                rows = cur.fetchall()
                items = [
                    {"judgment_id": r[0], "label": make_label(r[1], r[2]), "relationship": r[3]}
                    for r in rows
                ]
                return items, (int(rows[0][4]) if rows else 0)

            cites, cites_count = side("source_judgment_id", "target_judgment_id")
            cited_by, cited_by_count = side("target_judgment_id", "source_judgment_id")

        return {
            "judgment_id": judgment_id,
            "label": label,
            "cites_count": cites_count,
            "cited_by_count": cited_by_count,
            "cites": cites,
            "cited_by": cited_by,
        }

    # -- lookup ---------------------------------------------------------------

    def _resolver(self) -> CitationResolver:
        """Resolver rebuilt from judgment_aliases (or, before the migration,
        from judgment names), cached so /graph/resolve matches exactly like
        edge building without re-reading judgment text."""
        def build():
            with self._cursor() as cur:
                cur.execute("SELECT id, petitioner, respondent, date_of_judgment FROM judgments")
                judgments = cur.fetchall()
                years = {r[0]: _year_of(r[3]) for r in judgments}
                if self._table_exists("judgment_aliases"):
                    cur.execute("SELECT alias_norm, judgment_id, kind FROM judgment_aliases")
                    rows = cur.fetchall()
                    if rows:
                        return CitationResolver.from_alias_rows(rows, years)
            return CitationResolver(
                {"id": r[0], "petitioner": r[1], "respondent": r[2], "date_of_judgment": r[3]}
                for r in judgments
            )
        return self._cached("resolver", build)

    def resolve(self, citations: List[str]) -> List[Dict[str, Any]]:
        resolver = self._resolver()
        out = []
        for text in citations:
            res = resolver.resolve(text)
            out.append(
                {
                    "citation": text,
                    "judgment_id": res.judgment_id,
                    "method": res.method,
                    "score": round(res.score, 1),
                }
            )
        return out

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Cases matching a judgment ID, a reporter citation or (part of) a
        case name. Best matches first, each with how it matched."""
        q = (query or "").strip()
        if not q:
            return []
        hits: Dict[int, Tuple[str, float]] = {}

        def add(jid: int, match_type: str, score: float) -> None:
            if jid not in hits or hits[jid][1] < score:
                hits[jid] = (match_type, score)

        aliases = self._table_exists("judgment_aliases")
        with self._cursor() as cur:
            if q.isdigit():
                cur.execute("SELECT id FROM judgments WHERE id = %s", (int(q),))
                for (jid,) in cur.fetchall():
                    add(jid, "id", 100.0)

            reporters = parse_reporter_citations(q)
            if reporters and aliases:
                keys = [c.key for c in reporters] + [c.loose_key for c in reporters]
                cur.execute(
                    "SELECT DISTINCT judgment_id FROM judgment_aliases "
                    "WHERE kind = 'reporter' AND alias_norm = ANY(%s)",
                    (keys,),
                )
                for (jid,) in cur.fetchall():
                    add(jid, "reporter", 100.0)

            norm = normalize_case_name(q)
            if norm and not q.isdigit() and not reporters:
                like = "%" + norm.replace("%", "").replace("_", "\\_") + "%"
                if aliases and self._trigram_ready():
                    cur.execute(
                        "SELECT judgment_id, alias_norm, similarity(alias_norm, %s) AS sim "
                        "FROM judgment_aliases WHERE kind = 'case_name' "
                        "AND (alias_norm LIKE %s OR alias_norm %% %s) "
                        "ORDER BY (alias_norm = %s) DESC, (alias_norm LIKE %s) DESC, sim DESC "
                        "LIMIT %s",
                        (norm, like, norm, norm, like, limit),
                    )
                    for jid, alias, sim in cur.fetchall():
                        exact = alias == norm
                        add(jid, "exact_name" if exact else "name", 100.0 if exact else round(100 * float(sim), 1))
                elif aliases:
                    cur.execute(
                        "SELECT judgment_id, alias_norm FROM judgment_aliases "
                        "WHERE kind = 'case_name' AND alias_norm LIKE %s "
                        "ORDER BY length(alias_norm), judgment_id LIMIT %s",
                        (like, limit),
                    )
                    for jid, alias in cur.fetchall():
                        add(jid, "exact_name" if alias == norm else "name", 100.0 if alias == norm else 50.0)
                if not hits:
                    # Before the migration (or for names the aliases skip):
                    # match the raw judgment columns.
                    raw = "%" + q.replace("%", "").replace("_", "\\_") + "%"
                    cur.execute(
                        "SELECT id, petitioner, respondent FROM judgments "
                        "WHERE petitioner ILIKE %s OR respondent ILIKE %s ORDER BY id LIMIT %s",
                        (raw, raw, limit),
                    )
                    for jid, pet, res in cur.fetchall():
                        exact = case_name_alias(pet, res) == norm
                        add(jid, "exact_name" if exact else "name", 100.0 if exact else 50.0)

            ranked = sorted(hits.items(), key=lambda kv: (-kv[1][1], kv[0]))[:limit]
            info = self._judgments(cur, [jid for jid, _ in ranked])

        return [
            {
                "judgment_id": jid,
                "label": info.get(jid, {}).get("label", f"Case #{jid}"),
                "court": info.get(jid, {}).get("court"),
                "date": info.get(jid, {}).get("date"),
                "match_type": match_type,
                "score": score,
            }
            for jid, (match_type, score) in ranked
            if jid in info
        ]
