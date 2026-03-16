from src.retrieval.hybrid_retriever import retrieve

results = retrieve("What is vision transformer",5)

for r in results:
    print(r["paper_id"])
    print(r["text"][:200])
    print()