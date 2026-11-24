#!/usr/bin/env python3
"""Read-only data health check for the Legal RAG database.

Run from NLPv2/backend:
    python benchmark/db_health_check.py [--dsn postgresql://...] [--json out.json]

Only SELECTs. Tolerates missing tables/columns (older schemas). Prints a
report of metadata gaps, text quality, chunk/embedding integrity, duplicates
and citation-graph coverage.
"""
import argparse, json, sys
import psycopg2

DSN = "postgresql://postgres:postgres@localhost:5432/legal_rag"
R = {}


def conn_q(cur, sql, one=True):
    cur.execute(sql)
    rows = cur.fetchall()
    return rows[0] if (one and rows) else rows


def table_cols(cur):
    cur.execute("select table_name, column_name from information_schema.columns where table_schema='public'")
    d = {}
    for t, c in cur.fetchall():
        d.setdefault(t, set()).add(c)
    return d


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument("--json")
    a = ap.parse_args()
    con = psycopg2.connect(a.dsn)
    con.set_session(readonly=True, autocommit=True)
    cur = con.cursor()
    cur.execute("set statement_timeout = '1800s'")
    cols = table_cols(cur)
    J = cols.get("judgments", set())
    if not J:
        sys.exit("no judgments table")
    n = conn_q(cur, "select count(*) from judgments")[0]
    R["judgments_total"] = n
    print(f"judgments: {n}")

    def missing(name, cond):
        k = conn_q(cur, f"select count(*) from judgments where {cond}")[0]
        R[name] = k
        print(f"  {name:38s} {k:>8d}  {pct(k, n)}")

    print("\n== metadata gaps ==")
    for c in ("petitioner", "respondent", "court"):
        if c in J:
            missing(f"{c}_null_empty_unknown",
                    f"{c} is null or btrim({c})='' or lower(btrim({c})) in ('unknown','n/a','none','null')")
    if "date_of_judgment" in J:
        missing("date_null", "date_of_judgment is null")
        missing("date_before_1950", "date_of_judgment < '1950-01-26'")
        missing("date_in_future", "date_of_judgment > current_date")
        missing("date_is_jan1 (likely year-only/default)", "to_char(date_of_judgment,'MM-DD')='01-01'")
    if "bench" in J:
        missing("bench_empty", "bench is null or cardinality(bench)=0")
    if "citations" in J:
        missing("citations_empty", "citations is null or citations::text in ('[]','{}','null')")
    if "source_file" in J:
        missing("source_file_null", "source_file is null or btrim(source_file)=''")
    missing("judgment_text_null_empty", "judgment_text is null or btrim(judgment_text)=''")
    missing("judgment_text_lt_1000_chars", "length(judgment_text) < 1000")
    missing("judgment_text_gt_1M_chars", "length(judgment_text) > 1000000")

    print("\n== petitioner/respondent quality ==")
    if {"petitioner", "respondent"} <= J:
        missing("pet_or_resp_very_long(>200)", "length(petitioner)>200 or length(respondent)>200")
        missing("pet_or_resp_very_short(<3)", "length(btrim(petitioner))<3 or length(btrim(respondent))<3")
        missing("pet_or_resp_has_newline", "petitioner ~ E'\\n' or respondent ~ E'\\n'")
        missing("pet==resp", "lower(btrim(petitioner))=lower(btrim(respondent))")

    if "date_of_judgment" in J:
        print("\n== year distribution (decade) ==")
        rows = conn_q(cur, """select coalesce((extract(year from date_of_judgment)::int/10*10)::text,'NULL') d, count(*)
                              from judgments group by 1 order by 1""", one=False)
        R["decade_distribution"] = {r[0]: r[1] for r in rows}
        for d, c in rows:
            print(f"  {d:>6s} {c:>8d} {'#' * int(60 * c / max(1, n))}")
        # recoverable year for null-date rows
        print("\n== can missing dates be recovered? ==")
        r = conn_q(cur, """select
            count(*) filter (where date_of_judgment is null) nulls,
            count(*) filter (where date_of_judgment is null and substr(judgment_text,1,4000) ~ '(19[4-9][0-9]|20[0-2][0-9])') year_in_head,
            count(*) filter (where date_of_judgment is null and substr(judgment_text,1,4000) ~* '(\\d{1,2}(st|nd|rd|th)?[ ,]+(january|february|march|april|may|june|july|august|september|october|november|december)[ ,]+(19|20)\\d\\d)') full_date_in_head,
            count(*) filter (where date_of_judgment is null and coalesce(source_file,'') ~ '(19[4-9][0-9]|20[0-2][0-9])') year_in_filename
            from judgments""" if "source_file" in J else "select 0,0,0,0")
        R["date_recovery"] = dict(zip(["null", "year_in_text_head", "full_date_in_text_head", "year_in_filename"], r))
        print("  ", R["date_recovery"])

    print("\n== text quality (mojibake: non-ASCII ratio) ==")
    r = conn_q(cur, """with s as (select id, length(judgment_text) L,
            length(regexp_replace(substr(judgment_text,1,20000), '[\\x01-\\x7f]', '', 'g')) na
            from (select id, substr(judgment_text,1,20000) judgment_text from judgments where judgment_text is not null) q)
            select count(*) filter (where L>0 and na::float/L > 0.20),
                   count(*) filter (where L>0 and na::float/L > 0.05),
                   count(*) filter (where L>0 and na::float/L > 0.01) from s""")
    R["nonascii_gt20pct"], R["nonascii_gt5pct"], R["nonascii_gt1pct"] = r
    print(f"  >20% non-ASCII: {r[0]}  >5%: {r[1]}  >1%: {r[2]}")
    r = conn_q(cur, "select count(*) from judgments where judgment_text like '%' || chr(65533) || '%'")[0]
    R["has_replacement_char"] = r
    print(f"  contains U+FFFD: {r}")

    print("\n== duplicates ==")
    r = conn_q(cur, """select count(*), coalesce(sum(c-1),0) from (select count(*) c from judgments
                       group by md5(judgment_text) having count(*)>1) x""")
    R["dup_text_groups"], R["dup_text_extra_rows"] = r
    print(f"  identical judgment_text: {r[0]} groups, {r[1]} extra rows")
    if "source_file" in J:
        r = conn_q(cur, """select count(*), coalesce(sum(c-1),0) from (select count(*) c from judgments
                           where source_file is not null group by source_file having count(*)>1) x""")
        R["dup_source_groups"], R["dup_source_extra"] = r
        print(f"  same source_file: {r[0]} groups, {r[1]} extra rows")
    if {"petitioner", "respondent", "date_of_judgment"} <= J:
        r = conn_q(cur, """select count(*), coalesce(sum(c-1),0) from (select count(*) c from judgments
            where date_of_judgment is not null group by lower(petitioner), lower(respondent), date_of_judgment
            having count(*)>1) x""")
        R["dup_party_date_groups"] = r[0]
        print(f"  same petitioner+respondent+date: {r[0]} groups, {r[1]} extra rows (may be legit connected matters)")

    if "judgment_chunks" in cols:
        print("\n== chunks ==")
        r = conn_q(cur, "select count(*), count(distinct judgment_id), min(length(content)), percentile_disc(0.5) within group (order by length(content)), max(length(content)) from judgment_chunks")
        R["chunks"] = dict(zip(["total", "judgments_covered", "min_len", "median_len", "max_len"], r))
        print("  ", R["chunks"])
        k = conn_q(cur, "select count(*) from judgments j where not exists (select 1 from judgment_chunks c where c.judgment_id=j.id)")[0]
        R["judgments_without_chunks"] = k
        print(f"  judgments with NO chunks: {k} ({pct(k, n)})")
        k = conn_q(cur, "select count(*) from judgment_chunks where length(btrim(content)) < 50")[0]
        R["chunks_lt50"] = k
        print(f"  chunks < 50 chars: {k}")
        k = conn_q(cur, "select count(*) from judgment_chunks where length(regexp_replace(content,'[\\x01-\\x7f]','','g'))::float/greatest(length(content),1) > 0.2")[0]
        R["chunks_nonascii_gt20"] = k
        print(f"  chunks >20% non-ASCII: {k}")
        rows = conn_q(cur, "select section, count(*) from judgment_chunks group by 1 order by 2 desc", one=False)
        R["section_counts"] = dict(rows)
        print("  sections:", dict(rows))
        k = conn_q(cur, """select count(*) from (select judgment_id from judgment_chunks group by 1
                           having count(distinct section) = 1) x""")[0]
        R["judgments_with_single_section"] = k
        print(f"  judgments whose chunks are all one section: {k}")
        k = conn_q(cur, "select count(*) from judgment_chunks c where not exists (select 1 from judgments j where j.id=c.judgment_id)")[0]
        R["orphan_chunks"] = k
        print(f"  orphan chunks (no judgment): {k}")

    for emb in sorted(t for t in cols if t.endswith("embeddings") and t.startswith("judgment")):
        k = conn_q(cur, f"select count(*) from {emb}")[0]
        miss = conn_q(cur, f"select count(*) from judgment_chunks c where not exists (select 1 from {emb} e where e.chunk_id=c.chunk_id)")[0]
        orph = conn_q(cur, f"select count(*) from {emb} e where not exists (select 1 from judgment_chunks c where c.chunk_id=e.chunk_id)")[0]
        R[f"{emb}"] = dict(rows=k, chunks_missing_embedding=miss, orphan_embeddings=orph)
        print(f"\n== {emb} ==\n  rows {k}; chunks lacking embedding {miss}; orphan embeddings {orph}")

    if "citation_edges" in cols:
        print("\n== citation graph ==")
        E = cols["citation_edges"]
        tot = conn_q(cur, "select count(*) from citation_edges")[0]
        print(f"  edges: {tot}")
        R["edges"] = tot
        if {"source_judgment_id", "target_judgment_id"} <= E:
            r = conn_q(cur, """select
                count(*) filter (where target_judgment_id is null),
                count(*) filter (where source_judgment_id = target_judgment_id)
                from citation_edges""")
            R["edges_unresolved_target"], R["edges_self"] = r
            print(f"  unresolved target: {r[0]} ({pct(r[0], tot)}); self-edges: {r[1]}")
            k = conn_q(cur, "select count(*) from judgments j where not exists (select 1 from citation_edges e where e.source_judgment_id=j.id)")[0]
            k2 = conn_q(cur, "select count(*) from judgments j where not exists (select 1 from citation_edges e where e.target_judgment_id=j.id)")[0]
            R["judgments_no_outgoing"], R["judgments_never_cited"] = k, k2
            print(f"  judgments with no outgoing edges: {k} ({pct(k, n)}); never cited: {k2} ({pct(k2, n)})")
    else:
        print("\n(no citation_edges table: build_citation_edges.py not run / old schema)")
    if "judgment_aliases" in cols:
        k = conn_q(cur, "select count(distinct judgment_id) from judgment_aliases")[0]
        R["judgments_with_alias"] = k
        print(f"\naliases: {k} judgments have >=1 alias ({pct(k, n)})")
    if "judgment_graph_stats" in cols:
        k = conn_q(cur, "select count(*) from judgment_graph_stats")[0]
        print(f"graph stats rows: {k} ({pct(k, n)} of judgments)")
    if a.json:
        json.dump(R, open(a.json, "w"), indent=2, default=str)
        print("\nwrote", a.json)


if __name__ == "__main__":
    main()
