from typing import List, Dict, Any, Optional, Callable, Tuple


class PromptBuilder:

    SYSTEM_PROMPT = (
        "You are an Indian legal research assistant specializing in case law "
        "and statutes. Answer the question using the provided context. Each "
        "context item is numbered like [1], [2], etc. — cite the item number "
        "in brackets, e.g. [1], immediately after any claim you draw from it. "
        "Context items may be case law paragraphs or statutory provisions "
        "(Constitution articles, BNS sections, BNSS sections on criminal "
        "procedure). Synthesize across items as needed. When the question asks "
        "what to do or what happens next in a criminal matter, set out the "
        "procedure as ordered steps, citing the BNSS section for each. Only if the context is genuinely irrelevant to the question, "
        "respond with exactly: \"Not found in the provided cases or statutes.\""
    )

    NO_ANSWER_RESPONSE = "Not found in the provided cases or statutes."

    MIN_USEFUL_TOKENS = 40

    STATUTE_SECTION_HEADER = "## Relevant Statutes & Constitutional Provisions"
    CASE_LAW_SECTION_HEADER = "## Relevant Case Law"

    def __init__(self, max_context_tokens: int = 1200):
        self.max_context_tokens = max_context_tokens

    def build_context_block(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Backend-agnostic context block. Chunks are grouped by case/statute
        so the case metadata (name, court, year) or statute title is stated
        once per group, with that group's excerpts nested under it, instead
        of repeating the full citation on every chunk.

        Numbering [1..N] stays the FLAT, original 1-based index into
        `retrieved_chunks` even though chunks are visually grouped —
        post_processor._extract_citations maps [N] back to
        retrieved_chunks[N-1] by that exact positional index, so grouping
        must never renumber.

        Also injects a section header ("## Relevant Statutes & ...", then
        "## Relevant Case Law") whenever doc_type flips between consecutive
        groups. The pipeline always hands this method a statute-first,
        then-judgment-ordered list (see rag_pipeline.py), so in practice
        this fires at most twice and produces two clean sections — but an
        unsorted/mixed input would just re-flip the header at every
        boundary rather than break anything.
        """
        counter = token_counter or self._fallback_token_estimate
        budget = max_tokens if max_tokens is not None else self.max_context_tokens

        groups = self._group_chunks(retrieved_chunks)
        blocks: List[str] = []
        used = 0
        last_doc_type: Optional[str] = None

        for _key, header, members in groups:
            if budget - used <= self.MIN_USEFUL_TOKENS:
                break

            group_doc_type = members[0][1].get("doc_type", "judgment")
            if group_doc_type != last_doc_type:
                section_header = (
                    self.STATUTE_SECTION_HEADER
                    if group_doc_type == "statute"
                    else self.CASE_LAW_SECTION_HEADER
                )
                header_tokens = counter(section_header)
                if budget - used - header_tokens > self.MIN_USEFUL_TOKENS:
                    blocks.append(section_header)
                    used += header_tokens
                    last_doc_type = group_doc_type

            group_lines = [header]
            group_used = counter(header)
            for idx, chunk in members:
                remaining = budget - used - group_used
                if remaining <= self.MIN_USEFUL_TOKENS:
                    break

                excerpt = f"  [{idx}] {chunk['text']}"
                excerpt_tokens = counter(excerpt)
                if excerpt_tokens > remaining:
                    excerpt = self._truncate_to_tokens(excerpt, remaining, counter)
                    if not excerpt.strip():
                        break
                    excerpt_tokens = counter(excerpt)

                group_lines.append(excerpt)
                group_used += excerpt_tokens

            if len(group_lines) > 1:
                blocks.append("\n".join(group_lines))
                used += group_used

        return "\n\n".join(blocks)

    def _group_chunks(
        self, retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, List[Tuple[int, Dict[str, Any]]]]]:
        order: List[str] = []
        groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        headers: Dict[str, str] = {}

        for i, chunk in enumerate(retrieved_chunks, 1):
            key = self._group_key(chunk)
            if key not in groups:
                order.append(key)
                groups[key] = []
                headers[key] = self._group_header(chunk)
            groups[key].append((i, chunk))

        return [(k, headers[k], groups[k]) for k in order]

    def _group_key(self, chunk: Dict[str, Any]) -> str:
        if chunk.get("doc_type") == "statute":
            # statute_id isn't present on the chunk payload (retriever.py
            # drops it before returning the dict) — the statute title is
            # unique per statute in practice, so key on that instead.
            return f"statute:{chunk['case']}"
        return f"judgment:{chunk.get('judgment_id') or chunk['case']}"

    def _group_header(self, chunk: Dict[str, Any]) -> str:
        if chunk.get("doc_type") == "statute":
            return f"Statute: {chunk['case']}"
        return f"Case: {chunk['case']} ({chunk['court']}, {chunk['year']})"

    def _fallback_token_estimate(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def _truncate_to_tokens(self, text: str, budget: int, counter: Callable[[str], int]) -> str:
        if budget <= 0:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if counter(text[:mid] + "...") <= budget:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "..."

    def build_rag_prompt(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        user_query: str,
    ) -> str:
        """Deprecated — Qwen-specific string builder, superseded by
        build_context_block() + QwenBackend.generate(). Kept as a thin
        wrapper for demo.py and direct callers."""
        context = self.build_context_block(retrieved_chunks)
        return f"""<|im_start|>system
{self.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{user_query}<|im_end|>
<|im_start|>assistant
"""

    def get_citation_list(self, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        citations = []
        seen_cases = set()

        for chunk in retrieved_chunks:
            doc_type = chunk.get("doc_type", "judgment")
            sec_num = chunk.get("section_number") or chunk.get("para", "")
            if doc_type == "statute":
                case_key = f"{chunk['case']} {sec_num}"
            else:
                case_key = f"{chunk['case']} ({chunk['year']})"

            if case_key not in seen_cases:
                if doc_type == "statute":
                    citation = f"{chunk['case']} {sec_num}"
                else:
                    citation = f"{chunk['case']} ({chunk['court']}, {chunk['year']}, {chunk['para']})"
                citations.append(citation)
                seen_cases.add(case_key)

        return citations


if __name__ == "__main__":
    builder = PromptBuilder()

    sample_chunks = [
        {
            "doc_type": "judgment",
            "chunk_id": 1,
            "judgment_id": 101,
            "text": "The Supreme Court held that educational institutions must follow due process when implementing fee structures.",
            "case": "ABC University v. State",
            "court": "Supreme Court of India",
            "year": 2019,
            "para": "¶23",
            "section": "judgment",
            "similarity": 0.89,
        },
        {
            "doc_type": "judgment",
            "chunk_id": 2,
            "judgment_id": 101,
            "text": "The Court further clarified that fee revisions must be published in advance.",
            "case": "ABC University v. State",
            "court": "Supreme Court of India",
            "year": 2019,
            "para": "¶24",
            "section": "judgment",
            "similarity": 0.87,
        },
        {
            "doc_type": "judgment",
            "chunk_id": 3,
            "judgment_id": 202,
            "text": "The Karnataka Education Bill was challenged on constitutional grounds for violating fundamental rights.",
            "case": "Karnataka Students Association v. State",
            "court": "Karnataka High Court",
            "year": 2021,
            "para": "¶11",
            "section": "facts",
            "similarity": 0.85,
        },
    ]

    query = "What did the Supreme Court say about educational fees in Karnataka?"

    context = builder.build_context_block(sample_chunks)
    print("Generated context block:")
    print("=" * 50)
    print(context)

    print("\n" + "=" * 50)
    print("Citations:")
    for citation in builder.get_citation_list(sample_chunks):
        print(f"- {citation}")

    print(f"\nEstimated tokens: {builder._fallback_token_estimate(context):.0f}")
