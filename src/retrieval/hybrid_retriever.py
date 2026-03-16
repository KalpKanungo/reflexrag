import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_FILE = "indexes/vector_index.faiss"
METADATA_FILE = "indexes/metadata.pkl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE,"rb") as f:
    metadata = pickle.load(f)

texts = [m["text"] for m in metadata]

tokenized_corpus = [t.split() for t in texts]

bm25 = BM25Okapi(tokenized_corpus)

def retrieve(query,k=10):

    query_embedding = model.encode([query]).astype("float32")

    vector_scores,vector_ids = index.search(query_embedding,k)

    vector_scores = vector_scores[0]
    vector_ids = vector_ids[0]

    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = bm25_scores / np.max(bm25_scores)

    combined = []

    for i in range(len(metadata)):
        vscore = 0
        if i in vector_ids:
            pos = list(vector_ids).index(i)
            vscore = vector_scores[pos]

        bscore = bm25_scores[i]

        score = 0.6*vscore + 0.4*bscore

        combined.append((score,i))

    combined.sort(reverse=True)

    results = []

    for score,idx in combined[:k]:
        results.append(metadata[idx])

    return results