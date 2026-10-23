import streamlit as st

from lib import api_client
from lib.case_graph import case_label, render_case_graph_panel, render_path, selected_cases
from lib.mind_map import render_mind_map
from lib.network_view import render_network
from lib.ui_helpers import linked_label, render_api_error


st.set_page_config(page_title="Citation graph — Legal RAG", page_icon=":material/account_tree:", layout="wide")

st.title("Citation graph", icon=":material/account_tree:")
st.caption("Browse landmark cases and precedent networks derived from the judgment citation graph.")

NEIGHBOR_PAGE = 25
NETWORK_PAGE = 50
# Most cases per branch the mind map draws (the API's limit). Bigger
# neighbourhoods are pointed to the network table under it.
MIND_MAP_CAP = 50
MIND_MAP_MIN = 4
RELATIONSHIPS = ["overruled", "followed", "distinguished", "referred", "cited"]

st.session_state.setdefault("graph_focus", None)
st.session_state.setdefault("graph_trail", [])


def _focus(judgment_id: int, label: str) -> None:
    """Make a case the starting point for the ego graph and citation lists."""
    st.session_state["graph_focus"] = judgment_id
    trail = [t for t in st.session_state["graph_trail"] if t[0] != judgment_id]
    st.session_state["graph_trail"] = (trail + [(judgment_id, label)])[-8:]
    st.session_state["cites_offset"] = 0
    st.session_state["cited_by_offset"] = 0


def _focus_from(widget_key: str, options: dict, reset: bool = False) -> None:
    """Selectbox callback: focus the picked case. Runs only when the user
    changes the selection, so several pickers don't fight each other."""
    picked = st.session_state.get(widget_key)
    if picked in options:
        _focus(*options[picked])
        if reset:
            st.session_state[widget_key] = "—"


# "Map this case" links in the mind map open the page as Citation_Graph?focus=<id>.
try:
    _linked_focus = int(st.query_params.get("focus", ""))
except ValueError:
    _linked_focus = None
if _linked_focus and st.session_state.get("_linked_focus") != _linked_focus:
    st.session_state["_linked_focus"] = _linked_focus
    _focus(_linked_focus, f"Case #{_linked_focus}")


@st.cache_data(ttl=60, show_spinner=False)
def _search(q: str):
    return api_client.graph_search(q, limit=15)


@st.cache_data(ttl=300, show_spinner="Building mind map…")
def _mind_map(judgment_id: int, per_branch: int):
    return api_client.graph_mind_map(judgment_id, per_branch=per_branch)


@st.cache_data(ttl=60, show_spinner="Loading cases…")
def _neighbors(judgment_id: int, direction: str, offset: int):
    return api_client.graph_neighbors(judgment_id, direction=direction, limit=NEIGHBOR_PAGE, offset=offset)


@st.cache_data(ttl=300, show_spinner=False)
def _neighbor_totals(judgment_id: int) -> tuple:
    """How many cases a judgment cites and is cited by."""
    cites = api_client.graph_neighbors(judgment_id, direction="cites", limit=1)
    cited_by = api_client.graph_neighbors(judgment_id, direction="cited-by", limit=1)
    return cites["total"], cited_by["total"]


@st.cache_data(ttl=60, show_spinner=False)
def _network_page(judgment_id: int, direction: str, relationship, q, court, offset: int):
    return api_client.graph_neighbors(
        judgment_id, direction=direction, limit=NETWORK_PAGE, offset=offset, relationship=relationship, q=q, court=court
    )


def _graph_selected(cases, key: str) -> None:
    """Graph controls for the rows selected in a table above."""
    if cases:
        render_case_graph_panel(cases, key=key, pick=False)
    else:
        st.caption(":material/touch_app: Select rows in the table to draw their citation graph, full network "
                   "or the precedent path between two of them.")


def _network_table(judgment_id: int, n_cites: int, n_cited_by: int) -> None:
    """Every case around the judgment, searchable and filterable, for
    neighbourhoods too big to read off the mind map."""
    st.subheader("Network table")
    with st.container(horizontal=True):
        direction = st.segmented_control(
            "Direction",
            ["cited-by", "cites"],
            format_func=lambda d: f"Cited by ({n_cited_by:,})" if d == "cited-by" else f"Cites ({n_cites:,})",
            default="cited-by" if n_cited_by >= n_cites else "cites",
            key="net_direction",
        ) or "cited-by"
        q = st.text_input("Search", placeholder="Case name or citation", key="net_q").strip() or None
        relationship = st.selectbox("Relationship", ["All"] + RELATIONSHIPS, key="net_rel")
        relationship = None if relationship == "All" else relationship
        court = st.text_input("Court", placeholder="e.g. Supreme Court", key="net_court").strip() or None

    # Back to the first page whenever the case or a filter changes.
    filters = (judgment_id, direction, relationship, q, court)
    if st.session_state.get("net_filters") != filters:
        st.session_state["net_filters"] = filters
        st.session_state["net_offset"] = 0
    offset = st.session_state["net_offset"]

    try:
        with st.spinner("Loading cases…"):
            page = _network_page(judgment_id, direction, relationship, q, court, offset)
    except Exception as e:
        render_api_error(e)
        return
    total = page["total"]
    if not page["items"]:
        st.caption("No cases match these filters." if (q or relationship or court) else "None found.")
        return

    with st.spinner("Rendering table…"):
        event = st.dataframe(
            page["items"],
            column_order=["judgment_id", "label", "relationship", "court", "date", "pagerank_score", "cited_text"],
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            # A new page or filter is a new table: drop the old selection.
            key=f"net_table_{hash(filters)}_{offset}",
        )
    with st.container(horizontal=True):
        st.button(
            "Previous",
            key="net_prev",
            disabled=offset == 0,
            on_click=lambda: st.session_state.update(net_offset=max(0, offset - NETWORK_PAGE)),
        )
        st.caption(f"{offset + 1}–{offset + len(page['items'])} of {total:,}")
        st.button(
            "Next",
            key="net_next",
            disabled=offset + NETWORK_PAGE >= total,
            on_click=lambda: st.session_state.update(net_offset=offset + NETWORK_PAGE),
        )
    choices = {f"{i['label']} (#{i['judgment_id']})": (i["judgment_id"], i["label"]) for i in page["items"]}
    st.selectbox(
        "Map a case from this page",
        ["—"] + list(choices.keys()),
        key="net_open",
        on_change=_focus_from,
        args=("net_open", choices, True),
    )
    _graph_selected(selected_cases(page["items"], event), key="net_graph")


query = st.text_input(
    "Find a case",
    placeholder="Case name, citation (e.g. [1950] S.C.R. 940) or judgment ID",
)
if query.strip():
    try:
        results = _search(query.strip())
    except Exception as e:
        render_api_error(e)
        results = []
    if results:
        options = {
            f"{r['label']} (#{r['judgment_id']}, {r['date']})": (r["judgment_id"], r["label"]) for r in results
        }
        st.selectbox(
            f"{len(results)} matches",
            ["—"] + list(options.keys()),
            key="search_pick",
            on_change=_focus_from,
            args=("search_pick", options),
        )
    else:
        st.caption("No matching cases.")

tab_map, tab_landmarks, tab_neighbors, tab_path, tab_ego = st.tabs(
    ["Mind map", "Landmark cases", "Cites / Cited by", "Precedent path", "Full network"]
)

with tab_map:
    st.caption(
        "The case's precedents at a glance: what it cites and what cites it, grouped by how they were treated. "
        "Click a node to open or close it, click a case for its details."
    )
    controls = st.container(horizontal=True)
    with controls:
        map_id = st.number_input(
            "Judgment ID",
            min_value=1,
            value=st.session_state["graph_focus"],
            step=1,
            placeholder="Search above or type an ID",
            key=f"map_center_{st.session_state['graph_focus']}",
        )
    if map_id is None:
        st.caption("Search for a case above, or pick one from Landmark cases, to map its precedents.")
    else:
        try:
            n_cites, n_cited_by = _neighbor_totals(int(map_id))
        except Exception as e:
            render_api_error(e)
            n_cites = n_cited_by = None
        if n_cites is not None:
            # Only cases too small for a slider to change anything go without one.
            largest = max(n_cites, n_cited_by)
            if largest <= MIND_MAP_MIN:
                per_branch = max(largest, 1)
            else:
                with controls:
                    per_branch = st.slider(
                        "Cases per branch",
                        min_value=MIND_MAP_MIN,
                        max_value=MIND_MAP_CAP,
                        value=12,
                        help=(
                            f"This case cites {n_cites:,} cases and is cited by {n_cited_by:,}. The most important "
                            "on each side are shown; the rest are summarised as “+N more”."
                        ),
                    )
            if largest > MIND_MAP_CAP:
                st.info(
                    f"This case cites {n_cites:,} cases and is cited by {n_cited_by:,}. The mind map shows at most "
                    f"{MIND_MAP_CAP} per branch — use the network table below to search and filter all of them.",
                    icon=":material/table_rows:",
                )
            try:
                render_mind_map(_mind_map(int(map_id), per_branch), height=660)
            except Exception as e:
                render_api_error(e)
            _network_table(int(map_id), n_cites, n_cited_by)

with tab_landmarks:
    st.caption(
        "Top authority cases over the citation network. Citations per year divides a case's citations by "
        "the years since it was decided, so steadily cited recent cases aren't buried under old ones."
    )
    with st.container(horizontal=True):
        rank_by = st.segmented_control(
            "Rank by",
            ["pagerank", "citations_per_year", "in_degree"],
            format_func={
                "pagerank": "PageRank",
                "citations_per_year": "Citations per year",
                "in_degree": "Total citations",
            }.get,
            default="pagerank",
            key="landmark_rank_by",
        ) or "pagerank"
        limit = st.number_input("Limit", min_value=1, max_value=100, value=10)

    @st.cache_data(ttl=60)
    def _landmark_cases(limit: int, rank_by: str):
        return api_client.graph_landmark_cases(limit=limit, rank_by=rank_by)

    try:
        landmarks = _landmark_cases(limit, rank_by)
    except Exception as e:
        render_api_error(e)
        landmarks = []

    selected_judgment_id = None
    if landmarks:
        landmark_event = st.dataframe(
            landmarks,
            column_order=[
                "judgment_id", "label", "court", "date", "in_degree", "citations_per_year", "pagerank_score",
            ],
            width="stretch",
            on_select="rerun",
            selection_mode="multi-row",
            key=f"landmark_table_{limit}_{rank_by}",
        )
        options = {f"{c['label']} (#{c['judgment_id']})": c["judgment_id"] for c in landmarks}
        focus_options = {k: (v, k.rsplit(" (#", 1)[0]) for k, v in options.items()}
        picked = st.selectbox(
            "Focus a case from the table above",
            ["—"] + list(options.keys()),
            key="landmark_pick",
            on_change=_focus_from,
            args=("landmark_pick", focus_options),
        )
        if picked != "—":
            selected_judgment_id = options[picked]
        _graph_selected(selected_cases(landmarks, landmark_event), key="landmark_graph")
    else:
        st.caption("No landmark cases returned — citation graph may be empty.")

with tab_ego:
    st.caption("N-hop citation network centered on a single judgment. For large neighbourhoods the mind map is easier to read.")
    with st.container(horizontal=True):
        judgment_id = st.number_input(
            "Judgment ID",
            min_value=1,
            value=st.session_state["graph_focus"] or selected_judgment_id or 1,
            step=1,
        )
        depth = st.slider("Depth", min_value=1, max_value=3, value=2)
        max_nodes = st.number_input(
            "Max nodes",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Large graphs are slow to draw. Closest and most-cited cases are kept first.",
        )

    if st.button("Load ego graph", icon=":material/play_arrow:", type="primary"):
        try:
            with st.spinner("Loading network…"):
                data = api_client.graph_judgment(int(judgment_id), depth=depth, max_nodes=int(max_nodes))
            st.session_state["ego_graph"] = data
        except Exception as e:
            render_api_error(e)
            st.session_state.pop("ego_graph", None)

    # Kept in session state so the graph survives reruns from other widgets on the page.
    data = st.session_state.get("ego_graph")
    if data is not None:
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if not nodes:
            st.warning(f"Judgment {data.get('center_id')} not found in the citation graph.", icon=":material/search_off:")
        else:
            if data.get("truncated"):
                total = f"{data.get('total_nodes'):,}" + ("" if data.get("total_exact", True) else "+")
                st.info(
                    f"Showing the closest {len(nodes)} of {total} cases. Lower the depth or raise Max nodes to see more.",
                    icon=":material/filter_alt:",
                )
            render_network(data, height=660)
            with st.expander(f"Edges ({len(edges)})", icon=":material/list:"):
                st.dataframe(edges, width="stretch")

with tab_neighbors:
    st.caption("Cases this judgment cites and cases that cite it, most important first. Open one to go a level deeper.")
    trail = st.session_state["graph_trail"]
    if len(trail) > 1:
        st.caption("Trail: " + " → ".join(f"#{jid}" for jid, _ in trail))
    with st.container(horizontal=True):
        center_id = st.number_input(
            "Judgment ID",
            min_value=1,
            value=st.session_state["graph_focus"] or 1,
            step=1,
            # Keyed on the focus so the box follows it when a case is opened.
            key=f"neighbors_center_{st.session_state['graph_focus']}",
        )
        if st.button("Back", icon=":material/arrow_back:", disabled=len(trail) < 2):
            trail.pop()
            st.session_state["graph_focus"] = trail[-1][0]
            st.session_state["cites_offset"] = st.session_state["cited_by_offset"] = 0
            st.rerun()

    st.markdown(linked_label(f"Open judgment #{int(center_id)}", {"judgment_id": int(center_id)}))

    neighbor_selection = []
    col_cites, col_cited_by = st.columns(2)
    for col, direction, title in (
        (col_cites, "cites", "Cites"),
        (col_cited_by, "cited-by", "Cited by"),
    ):
        offset_key = f"{direction.replace('-', '_')}_offset"
        offset = st.session_state.get(offset_key, 0)
        with col:
            try:
                page = _neighbors(int(center_id), direction, offset)
            except Exception as e:
                render_api_error(e)
                continue
            total = page["total"]
            st.subheader(f"{title} ({total})")
            if not page["items"]:
                st.caption("None found.")
                continue
            event = st.dataframe(
                page["items"],
                column_order=["judgment_id", "label", "relationship", "date", "pagerank_score"],
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key=f"{direction}_table_{int(center_id)}_{offset}",
            )
            neighbor_selection += selected_cases(page["items"], event)
            with st.container(horizontal=True):
                if st.button("Previous", key=f"{direction}_prev", disabled=offset == 0):
                    st.session_state[offset_key] = max(0, offset - NEIGHBOR_PAGE)
                    st.rerun()
                st.caption(f"{offset + 1}–{offset + len(page['items'])} of {total}")
                if st.button("Next", key=f"{direction}_next", disabled=offset + NEIGHBOR_PAGE >= total):
                    st.session_state[offset_key] = offset + NEIGHBOR_PAGE
                    st.rerun()
            choices = {f"{i['label']} (#{i['judgment_id']})": (i["judgment_id"], i["label"]) for i in page["items"]}
            st.selectbox(
                "Open a case",
                ["—"] + list(choices.keys()),
                key=f"{direction}_open",
                on_change=_focus_from,
                args=(f"{direction}_open", choices, True),
            )
    # Selections from both tables, without repeats.
    _graph_selected(list(dict(neighbor_selection).items()), key="neighbor_graph")

with tab_path:
    st.caption(
        "Shortest citation chain between two judgments — each case in the chain cites the next. "
        "Both directions are tried, so the order doesn't matter."
    )
    with st.container(horizontal=True):
        source_id = st.number_input(
            "From judgment ID", min_value=1, value=st.session_state["graph_focus"], step=1,
            placeholder="Search above or type an ID", key=f"src_{st.session_state['graph_focus']}",
        )
        target_id = st.number_input("To judgment ID", min_value=1, value=None, step=1,
                                    placeholder="Type an ID", key="tgt")
    ends = [int(i) for i in (source_id, target_id) if i is not None]
    if ends:
        st.caption(" ↔ ".join(f"{case_label(i)} (#{i})" for i in ends))

    if st.button("Find path", icon=":material/route:", type="primary", disabled=len(ends) < 2):
        st.session_state["path_ends"] = tuple(ends)
    # Kept in session state so the result survives reruns from other widgets on the page.
    path_ends = st.session_state.get("path_ends")
    if path_ends:
        render_path(*((jid, case_label(jid)) for jid in path_ends))
