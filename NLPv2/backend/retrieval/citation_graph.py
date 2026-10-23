"""
Citation Graph Manager for Legal RAG

Serves the directed precedent graph: subgraphs for visual APIs, landmark
cases, precedent chains, PageRank / centrality scores, search and citation
resolution.

Two modes, same public methods and return shapes:
  - "postgres" (default): indexed queries against citation_edges and the
    precomputed judgment_graph_stats table (see graph/pg_store.py). Nothing
    is held in memory, so it scales with the corpus and needs no reload.
  - "memory": the original approach, the whole graph loaded into NetworkX.
    Used when GRAPH_BACKEND=memory, or when a caller fills `self.graph`
    itself and sets `_is_loaded` (as the smoke tests do).
"""

import os
from typing import Dict, List, Any, Optional
import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor

from graph.citation_resolver import _year_of
from graph.pg_store import LANDMARK_RANKINGS, PostgresGraphStore, citations_per_year, rank_landmarks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class CitationGraphManager:
    """Citation graph queries, backed by PostgreSQL (default) or an in-memory
    NetworkX copy of it."""

    def __init__(
        self,
        db_host: Optional[str] = None,
        db_port: Optional[str] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.db_host = db_host or os.getenv("DB_HOST", "localhost")
        # 5433 matches retriever.py's default and the docker-compose host port
        # mapping — keep these in sync, they drifted apart once already.
        self.db_port = db_port or os.getenv("DB_PORT", "5433")
        self.db_name = db_name or os.getenv("DB_NAME", "legal_rag")
        self.db_user = db_user or os.getenv("DB_USER", "postgres")
        self.db_password = db_password or os.getenv("DB_PASSWORD", "postgres")

        self.graph = nx.DiGraph()
        self._is_loaded = False
        self._pagerank_cache: Optional[tuple] = None

        self.backend = (backend or os.getenv("GRAPH_BACKEND", "postgres")).lower()
        self._pg = PostgresGraphStore(
            {
                "host": self.db_host,
                "port": self.db_port,
                "dbname": self.db_name,
                "user": self.db_user,
                "password": self.db_password,
            }
        )

    def _use_postgres(self) -> bool:
        # A graph someone loaded or filled in memory always wins.
        return self.backend != "memory" and not self._is_loaded

    def _get_connection(self):
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def _mem_load(self, force_reload: bool = False) -> int:
        """
        Load nodes and edges from PostgreSQL into NetworkX.
        Returns total number of edges loaded.
        """
        if self._is_loaded and not force_reload:
            return self.graph.number_of_edges()

        self.graph.clear()

        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Add judgment nodes with attributes
                cur.execute(
                    "SELECT id, petitioner, respondent, court, date_of_judgment FROM judgments;"
                )
                judgments = cur.fetchall()
                for j in judgments:
                    petitioner = j.get("petitioner") or "Unknown"
                    respondent = j.get("respondent") or "Unknown"
                    label = f"{petitioner} v. {respondent}"
                    self.graph.add_node(
                        j["id"],
                        label=label,
                        court=j.get("court"),
                        date=str(j.get("date_of_judgment")),
                    )

                # Add citation edges
                cur.execute(
                    "SELECT source_judgment_id, target_judgment_id, cited_text, relationship_type FROM citation_edges;"
                )
                edges = cur.fetchall()
                for e in edges:
                    src = e["source_judgment_id"]
                    tgt = e["target_judgment_id"]
                    if tgt is not None and self.graph.has_node(tgt):
                        self.graph.add_edge(
                            src,
                            tgt,
                            cited_text=e["cited_text"],
                            relationship=e["relationship_type"],
                        )

            conn.close()
            self._is_loaded = True
        except Exception as err:
            print(f"Warning: Failed to load citation graph from DB: {err}")

        return self.graph.number_of_edges()

    def _mem_subgraph(
        self, judgment_id: int, depth: int = 2, max_nodes: int = 100
    ) -> Dict[str, Any]:
        """
        Extract an N-hop ego graph around a specific judgment ID, capped at
        max_nodes. Returns JSON-serializable dict of nodes and edges.

        Around well-cited judgments a depth-3 neighbourhood can run into
        thousands of nodes, which the frontend cannot lay out. When the cap is
        hit, nodes closer to the center are kept first and, within a hop
        level, the most-connected ones win. Since every hop level is kept in
        full before the next one is touched, the result stays connected.
        """
        self._mem_load()

        if judgment_id not in self.graph:
            return {
                "nodes": [],
                "edges": [],
                "center_id": judgment_id,
                "total_nodes": 0,
                "total_exact": True,
                "truncated": False,
            }

        dist = nx.single_source_shortest_path_length(
            self.graph.to_undirected(as_view=True), judgment_id, cutoff=depth
        )
        total_nodes = len(dist)
        truncated = total_nodes > max_nodes
        if truncated:
            keep = sorted(dist, key=lambda n: (dist[n], -self.graph.degree(n), n))[:max_nodes]
        else:
            keep = list(dist)

        ego_g = self.graph.subgraph(keep)

        nodes = []
        for n, data in ego_g.nodes(data=True):
            nodes.append(
                {
                    "id": n,
                    "label": data.get("label", f"Case #{n}"),
                    "court": data.get("court"),
                    "date": data.get("date"),
                    "is_center": (n == judgment_id),
                    "hop": dist[n],
                }
            )

        edges = []
        for u, v, data in ego_g.edges(data=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "cited_text": data.get("cited_text", ""),
                    "relationship": data.get("relationship", "cited"),
                }
            )

        return {
            "center_id": judgment_id,
            "nodes": nodes,
            "edges": edges,
            "total_nodes": total_nodes,
            "total_exact": True,
            "truncated": truncated,
        }

    def _mem_landmark_cases(self, limit: int = 10, rank_by: str = "pagerank") -> List[Dict[str, Any]]:
        """
        Identify top authority cases by PageRank, citations per year or
        in-degree (see graph/pg_store.py for how citations per year is counted).
        """
        self._mem_load()

        if self.graph.number_of_nodes() == 0:
            return []

        pagerank_scores = nx.pagerank(self.graph) if self.graph.number_of_edges() > 0 else {}
        years = {n: _year_of(d.get("date")) for n, d in self.graph.nodes(data=True)}
        end_year = max((y for y in years.values() if y), default=None)

        rows = [
            (
                node,
                pagerank_scores.get(node, 0.0),
                self.graph.in_degree(node),
                citations_per_year(self.graph.in_degree(node), years[node], end_year),
            )
            for node in self.graph.nodes()
        ]
        landmarks = []
        for node, score, in_degree, cpy in rank_landmarks(rows, rank_by, limit):
            data = self.graph.nodes[node]
            landmarks.append(
                {
                    "judgment_id": node,
                    "label": data.get("label", f"Case #{node}"),
                    "court": data.get("court"),
                    "date": data.get("date"),
                    "in_degree": in_degree,
                    "citations_per_year": round(cpy, 3),
                    "pagerank_score": round(score, 6),
                }
            )
        return landmarks

    def _mem_shortest_path(self, source_id: int, target_id: int) -> Optional[List[int]]:
        """
        Find shortest citation chain between source and target judgments.
        """
        self._mem_load()
        if source_id not in self.graph or target_id not in self.graph:
            return None

        # Same date pruning as PostgresGraphStore.shortest_path: a chain only
        # runs back in time, so it stays between the target's date and the source's.
        def date_of(n: int) -> Optional[str]:
            d = str(self.graph.nodes[n].get("date") or "")
            return d[:10] if len(d) >= 10 and d[4] == "-" else None

        newest, oldest = date_of(source_id), date_of(target_id)
        if newest and oldest and newest < oldest:
            return None

        def in_window(n: int) -> bool:
            d = date_of(n)
            return d is None or ((not oldest or d >= oldest) and (not newest or d <= newest))

        try:
            return nx.bidirectional_shortest_path(
                nx.subgraph_view(self.graph, filter_node=in_window), source_id, target_id
            )
        except nx.NetworkXNoPath:
            return None

    def _mem_centrality_scores(self) -> Dict[int, float]:
        """
        Return normalized PageRank scores for all judgments (used for retrieval boosting).
        """
        self._mem_load()
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            return {}
        # Cached per graph size: retrieval asks for this on every query.
        shape = (self.graph.number_of_nodes(), self.graph.number_of_edges())
        if self._pagerank_cache is None or self._pagerank_cache[0] != shape:
            self._pagerank_cache = (shape, nx.pagerank(self.graph))
        return self._pagerank_cache[1]

    def _mem_precedent_summary(
        self, judgment_id: int, top_n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Compact precedent-chain summary for a judgment: how many cases it cites,
        how many cases cite it, and the strongest precedents followed/applied
        later. Returns None if the judgment is not in the graph.
        """
        self._mem_load()
        if judgment_id not in self.graph:
            return None

        out_citations = []
        in_citations = []

        for _src, tgt, data in self.graph.out_edges(judgment_id, data=True):
            out_citations.append(
                {
                    "judgment_id": tgt,
                    "label": self.graph.nodes[tgt].get("label", f"Case #{tgt}"),
                    "relationship": data.get("relationship", "cited"),
                }
            )

        for src, _tgt, data in self.graph.in_edges(judgment_id, data=True):
            in_citations.append(
                {
                    "judgment_id": src,
                    "label": self.graph.nodes[src].get("label", f"Case #{src}"),
                    "relationship": data.get("relationship", "cited"),
                }
            )

        rel_rank = {
            "overruled": 0,
            "followed": 1,
            "distinguished": 2,
            "referred": 3,
            "cited": 4,
        }
        in_citations.sort(
            key=lambda c: (rel_rank.get(c["relationship"], 5), c["judgment_id"])
        )
        out_citations.sort(
            key=lambda c: (rel_rank.get(c["relationship"], 5), c["judgment_id"])
        )

        return {
            "judgment_id": judgment_id,
            "label": self.graph.nodes[judgment_id].get("label", f"Case #{judgment_id}"),
            "cites_count": len(out_citations),
            "cited_by_count": len(in_citations),
            "cites": out_citations[:top_n],
            "cited_by": in_citations[:top_n],
        }

    def _mem_neighbors(
        self,
        judgment_id: int,
        direction: str,
        relationship: Optional[str],
        limit: int,
        offset: int,
        q: Optional[str] = None,
        court: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self._mem_load()
        if judgment_id not in self.graph:
            return None
        edges = (
            self.graph.out_edges(judgment_id, data=True)
            if direction == "cites"
            else self.graph.in_edges(judgment_id, data=True)
        )
        scores = self._mem_centrality_scores()
        items = []
        for src, tgt, data in edges:
            other = tgt if direction == "cites" else src
            rel = data.get("relationship", "cited")
            if other == judgment_id or (relationship and rel != relationship):
                continue
            node = self.graph.nodes[other]
            label = node.get("label", f"Case #{other}")
            cited_text = data.get("cited_text", "")
            if q and q.lower() not in label.lower() and q.lower() not in cited_text.lower():
                continue
            if court and court.lower() not in (node.get("court") or "").lower():
                continue
            items.append(
                {
                    "judgment_id": other,
                    "label": label,
                    "court": node.get("court"),
                    "date": node.get("date"),
                    "relationship": rel,
                    "cited_text": cited_text,
                    "pagerank_score": round(scores.get(other, 0.0), 6),
                }
            )
        items.sort(key=lambda i: (-i["pagerank_score"], i["judgment_id"]))
        return {
            "judgment_id": judgment_id,
            "direction": direction,
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "items": items[offset : offset + limit],
        }

    # ------------------------------------------------------------------
    # Public API — same names, arguments and return shapes in both modes.
    # ------------------------------------------------------------------

    def load_graph_from_db(self, force_reload: bool = False) -> int:
        """
        Memory mode: load nodes and edges from PostgreSQL into NetworkX.
        Postgres mode: nothing to load; checks the connection.
        Returns the number of (distinct) citation edges.
        """
        if not self._use_postgres():
            return self._mem_load(force_reload)
        try:
            return self._pg.edge_count()
        except Exception as err:
            print(f"Warning: Failed to reach citation graph in DB: {err}")
            return 0

    def get_subgraph(
        self, judgment_id: int, depth: int = 2, max_nodes: int = 100
    ) -> Dict[str, Any]:
        """
        N-hop ego graph around a judgment, capped at max_nodes (closest
        nodes first, then the most important). Returns nodes, edges,
        total_nodes and truncated.
        """
        if self._use_postgres():
            return self._pg.subgraph(judgment_id, depth=depth, max_nodes=max_nodes)
        return self._mem_subgraph(judgment_id, depth=depth, max_nodes=max_nodes)

    def get_landmark_cases(self, limit: int = 10, rank_by: str = "pagerank") -> List[Dict[str, Any]]:
        """Top authority cases by rank_by: "pagerank" (default),
        "citations_per_year" or "in_degree"."""
        if rank_by not in LANDMARK_RANKINGS:
            raise ValueError(f"rank_by must be one of {LANDMARK_RANKINGS}")
        if self._use_postgres():
            return self._pg.landmark_cases(limit=limit, rank_by=rank_by)
        return self._mem_landmark_cases(limit=limit, rank_by=rank_by)

    def get_shortest_path(self, source_id: int, target_id: int) -> Optional[List[int]]:
        """Shortest citation chain from source to target, or None."""
        if self._use_postgres():
            return self._pg.shortest_path(source_id, target_id)
        return self._mem_shortest_path(source_id, target_id)

    def get_centrality_scores(self) -> Dict[int, float]:
        """PageRank for every judgment with a non-zero score (retrieval boosting)."""
        if self._use_postgres():
            try:
                return self._pg.all_scores()
            except Exception as err:
                print(f"Warning: centrality scores unavailable: {err}")
                return {}
        return self._mem_centrality_scores()

    def centrality_for(self, judgment_ids: List[int]) -> Dict[int, float]:
        """PageRank for just these judgments — what retrieval needs per query,
        without building the full map."""
        ids = [i for i in dict.fromkeys(judgment_ids) if i is not None]
        if self._use_postgres():
            try:
                return self._pg.scores_for(ids)
            except Exception as err:
                print(f"Warning: centrality scores unavailable: {err}")
                return {}
        scores = self._mem_centrality_scores()
        return {i: scores[i] for i in ids if i in scores}

    def get_precedent_summary(
        self, judgment_id: int, top_n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Compact precedent-chain summary for a judgment: how many cases it
        cites, how many cite it, and the strongest of each. None if the
        judgment is not in the graph.
        """
        if self._use_postgres():
            try:
                return self._pg.precedent_summary(judgment_id, top_n=top_n)
            except Exception as err:
                # Answering a query must not fail because the graph is unavailable.
                print(f"Warning: precedent summary unavailable: {err}")
                return None
        return self._mem_precedent_summary(judgment_id, top_n=top_n)

    def get_neighbors(
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
        cite it ("cited_by"), most important first. q filters on the case name
        or citation text and court on the court name (case-insensitive
        substrings). None if it doesn't exist."""
        if direction not in ("cites", "cited_by"):
            raise ValueError("direction must be 'cites' or 'cited_by'")
        if self._use_postgres():
            return self._pg.neighbors(judgment_id, direction, relationship, limit, offset, q, court)
        return self._mem_neighbors(judgment_id, direction, relationship, limit, offset, q, court)

    # Display order for relationship groups in the mind map, strongest first.
    RELATIONSHIP_ORDER = ["overruled", "followed", "distinguished", "referred", "cited"]

    def _judgment_info(self, judgment_id: int) -> Dict[str, Any]:
        """Label, court and date of one judgment."""
        if self._use_postgres():
            return self._pg.judgment_info(judgment_id)
        data = self.graph.nodes[judgment_id]
        return {"label": data.get("label", f"Case #{judgment_id}"), "court": data.get("court"), "date": data.get("date")}

    def get_mind_map(
        self, judgment_id: int, per_branch: int = 12, child_per_branch: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        A judgment's precedent neighbourhood as a tree, for a mind-map view:
        root case -> "Cites" / "Cited by" -> relationship groups -> the most
        important cases -> each of those cases' own top citations. Lists are
        capped (most important first) and say how many were left out, so the
        tree stays readable around landmark cases with thousands of
        citations. None if the judgment doesn't exist.
        """
        cites = self.get_neighbors(judgment_id, "cites", limit=per_branch)
        if cites is None:
            return None
        cited_by = self.get_neighbors(judgment_id, "cited_by", limit=per_branch)

        def case(item: Dict[str, Any], **extra) -> Dict[str, Any]:
            return {
                "kind": "case",
                "id": item["judgment_id"],
                "label": item["label"],
                "court": item.get("court"),
                "date": str(item["date"]) if item.get("date") else None,
                "relationship": item.get("relationship"),
                **extra,
            }

        def more(shown: int, total: int) -> List[Dict[str, Any]]:
            if total <= shown:
                return []
            return [{"kind": "more", "label": f"+{total - shown} more", "count": total - shown}]

        def branch_node(direction: str, total: int, children: List[Dict[str, Any]]) -> Dict[str, Any]:
            label = "Cites" if direction == "cites" else "Cited by"
            return {"kind": "branch", "direction": direction, "label": label, "count": total, "children": children}

        def child_branches(item: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Second level: a neighbour's own top citations, ungrouped. The
            root is left out, since it is already on screen."""
            branches = []
            for direction in ("cites", "cited_by"):
                page = self.get_neighbors(item["judgment_id"], direction, limit=child_per_branch + 1)
                if not page:
                    continue
                items = [i for i in page["items"] if i["judgment_id"] != judgment_id][:child_per_branch]
                total = page["total"] - any(i["judgment_id"] == judgment_id for i in page["items"])
                if total:
                    branches.append(branch_node(direction, total, [case(i) for i in items] + more(len(items), total)))
            return branches

        def top_branch(page: Dict[str, Any], direction: str) -> Dict[str, Any]:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for item in page["items"]:
                rel = item.get("relationship") or "cited"
                groups.setdefault(rel, []).append(case(item, children=child_branches(item)))
            order = {r: n for n, r in enumerate(self.RELATIONSHIP_ORDER)}
            rels = sorted(groups, key=lambda r: (order.get(r, len(order)), r))
            if len(rels) == 1:
                # A single kind of relationship: a group level would only add a click.
                children = groups[rels[0]]
            else:
                children = [
                    {"kind": "group", "label": rel.capitalize(), "count": len(groups[rel]), "children": groups[rel]}
                    for rel in rels
                ]
            return branch_node(direction, page["total"], children + more(len(page["items"]), page["total"]))

        return {
            "kind": "case",
            "id": judgment_id,
            **self._judgment_info(judgment_id),
            "children": [top_branch(cites, "cites"), top_branch(cited_by, "cited_by")],
        }

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Cases matching an ID, a reporter citation or a case name. Always
        queries PostgreSQL, in either mode."""
        return self._pg.search(query, limit=limit)

    def resolve_citations(self, citations: List[str]) -> List[Dict[str, Any]]:
        """Resolve citation strings to judgment IDs, exactly as edge building
        does. Always queries PostgreSQL, in either mode."""
        return self._pg.resolve(citations)
