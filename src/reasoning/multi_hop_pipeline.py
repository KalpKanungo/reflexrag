from src.retrieval.hybrid_retriever import retrieve
from src.retrieval.reranker import Reranker

class MultiHopRetriever:
    def __init__(self, reranker_model_path):
        self.reranker = Reranker(reranker_model_path)

    def decompose_query(self, query):
        query = query.lower()

        sub_queries = [query]

        if "why" in query or "better" in query or "difference" in query:
            if " and " in query:
                parts = query.split(" and ")
                sub_queries.extend(parts)
            elif " vs " in query:
                parts = query.split(" vs ")
                sub_queries.extend(parts)
            else:
                words = query.split()
                if len(words) > 4:
                    sub_queries.append(" ".join(words[:len(words)//2]))
                    sub_queries.append(" ".join(words[len(words)//2:]))

        if "how" in query:
            sub_queries.append(query.replace("how", "method"))
            sub_queries.append(query.replace("how", "approach"))

        if "what" in query:
            sub_queries.append(query.replace("what", "definition"))

        return list(dict.fromkeys([q.strip() for q in sub_queries if len(q) > 5]))

    def retrieve_multi_hop(self, query, top_k=5):
        sub_queries = self.decompose_query(query)

        all_results = []

        for sub_q in sub_queries:
            retrieved = retrieve(sub_q, k=20)
            reranked = self.reranker.rerank(sub_q, retrieved, top_k=top_k)

            for r in reranked:
                r["source_query"] = sub_q
                all_results.append(r)

        unique = {}
        for r in all_results:
            text = r["text"]
            if text not in unique or r["score"] > unique[text]["score"]:
                unique[text] = r

        final = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

        return final[:top_k]