import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.reasoning.query_decomposition import QueryDecomposer
from src.retrieval.reranker import Reranker
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
index = faiss.read_index("indexes/vector_index.faiss")

with open("indexes/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def retrieve(query, top_k=20):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        results.append(metadata[i])
    return results

def main():
    query = input("Enter your query: ")

    decomposer = QueryDecomposer()
    sub_queries = decomposer.decompose(query)

    print("\nSub-queries:\n")
    for q in sub_queries:
        print("-", q)

    seen = set()
    all_results = []
    for sub_q in sub_queries:
        results = retrieve(sub_q, top_k=10)
        for r in results:
            key = (r["paper_id"], r["text"][:50])
            if key not in seen:
                seen.add(key)
                all_results.append(r)

    print(f"\nTotal unique chunks retrieved: {len(all_results)}")

    reranker = Reranker("models/reranker_finetuned")
    reranked = reranker.rerank(query, all_results, top_k=5)

    print("\nFinal Top Results:\n")
    for r in reranked:
        print("Paper:", r["paper_id"])
        print("Score:", r["score"])
        print("Text:", r["text"][:300])
        print("-" * 80)

if __name__ == "__main__":
    main()