from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 8
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        
        reranked = sorted(
            chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        return reranked[:top_n]

    def rerank_keep_pinned(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 8,
    ) -> List[Dict[str, Any]]:
        """Rerank, but keep exact-match rows (an explicit "section N"
        statute, or a judgment the query names by citation or title) first
        and unscored: the cross-encoder would otherwise re-sort a guaranteed
        match out of the final top_n."""
        pinned = [c for c in chunks if c.get("exact_match")][:top_n]
        rest = [c for c in chunks if not c.get("exact_match")]
        return pinned + self.rerank(query, rest, top_n=top_n - len(pinned))
