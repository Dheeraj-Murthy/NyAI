"""
Smoke Test for Citation Graph & NetworkX Integration
"""

import sys
import os
import networkx as nx

# Add backend/ (parent of this tests/ dir, for the retrieval package) and
# ingestion/ (sibling of backend/) to path.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "..", "ingestion"))

from citation_extractor import CitationExtractor
from retrieval.citation_graph import CitationGraphManager


def test_citation_extractor():
    print("--- Testing CitationExtractor ---")
    extractor = CitationExtractor()
    sample_text = (
        "In the landmark decision of Vijay Kumar v. State of Karnataka (1987), "
        "the court established key principles. This was later overruled in AIR 1992 SC 588."
    )
    extracted = extractor.extract_citations(sample_text)
    print(f"Extracted {len(extracted)} citations from sample text:")
    for item in extracted:
        print(f"  - Cited Text: {item['cited_text']} | Relationship: {item['relationship_type']}")

    assert len(extracted) >= 1, "Should extract at least 1 citation"
    print("✓ CitationExtractor test passed!")


def test_citation_graph_manager():
    print("\n--- Testing CitationGraphManager (Synthetic Graph) ---")
    gm = CitationGraphManager()

    # Manually populate graph nodes & edges to test in-memory algorithms
    gm.graph.add_node(1, label="Kesavananda Bharati v. State of Kerala", court="Supreme Court", date="1973-04-24")
    gm.graph.add_node(2, label="Maneka Gandhi v. Union of India", court="Supreme Court", date="1978-01-25")
    gm.graph.add_node(3, label="Minerva Mills v. Union of India", court="Supreme Court", date="1980-07-31")

    gm.graph.add_edge(2, 1, relationship="followed", cited_text="Kesavananda Bharati case")
    gm.graph.add_edge(3, 1, relationship="followed", cited_text="AIR 1973 SC 1461")
    gm.graph.add_edge(3, 2, relationship="referred", cited_text="Maneka Gandhi judgment")
    gm._is_loaded = True

    # Test Subgraph
    subgraph = gm.get_subgraph(1, depth=2)
    print(f"Subgraph for Case #1: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
    assert len(subgraph["nodes"]) == 3, "Subgraph should contain 3 connected nodes"

    # Test Landmark Cases (PageRank)
    landmarks = gm.get_landmark_cases(limit=3)
    print(f"Landmark cases (top authority):")
    for lm in landmarks:
        print(f"  - [{lm['judgment_id']}] {lm['label']} (PageRank: {lm['pagerank_score']}, In-Degree: {lm['in_degree']})")
    assert landmarks[0]["judgment_id"] == 1, "Case #1 should have highest PageRank/in-degree"

    # Citations per year: #1 (1973) cited twice over 8 years to 1980 = 0.25;
    # #2 (1978) cited once, floored at 5 years = 0.2.
    by_rate = gm.get_landmark_cases(limit=3, rank_by="citations_per_year")
    assert [lm["judgment_id"] for lm in by_rate[:2]] == [1, 2], by_rate
    assert by_rate[0]["citations_per_year"] == 0.25 and by_rate[1]["citations_per_year"] == 0.2, by_rate
    assert gm.get_landmark_cases(limit=1, rank_by="in_degree")[0]["judgment_id"] == 1
    try:
        gm.get_landmark_cases(rank_by="popularity")
        raise AssertionError("an unknown rank_by should be rejected")
    except ValueError:
        pass

    # Test Shortest Path
    path = gm.get_shortest_path(3, 1)
    print(f"Shortest path from Case #3 -> Case #1: {path}")
    assert path == [3, 1], "Path should be direct edge [3, 1]"
    assert gm.get_shortest_path(1, 3) is None, "An older case can't lead to a newer one"

    # Test Mind Map
    tree = gm.get_mind_map(1, per_branch=1)
    cites, cited_by = tree["children"]
    print(f"Mind map for Case #1: cites {cites['count']}, cited by {cited_by['count']}")
    assert tree["label"].startswith("Kesavananda") and cites["count"] == 0
    assert cited_by["count"] == 2 and [c["kind"] for c in cited_by["children"]] == ["case", "more"]
    case3 = gm.get_mind_map(1)["children"][1]["children"][1]
    assert case3["id"] == 3 and [b["count"] for b in case3["children"]] == [1], "Root is left out of a child's branches"
    assert gm.get_mind_map(99) is None

    # Test neighbour search / filters (the mind map's network table)
    assert gm.get_neighbors(1, "cited_by", q="gandhi v. union")["total"] == 1
    assert gm.get_neighbors(1, "cited_by", q="1973 sc")["items"][0]["judgment_id"] == 3, "q matches citation text"
    assert gm.get_neighbors(1, "cited_by", court="supreme")["total"] == 2
    assert gm.get_neighbors(1, "cited_by", q="nothing like this")["total"] == 0

    print("✓ CitationGraphManager synthetic test passed!")


if __name__ == "__main__":
    test_citation_extractor()
    test_citation_graph_manager()
    print("\n✓ ALL SMOKE TESTS PASSED SUCCESSFULLY!")
