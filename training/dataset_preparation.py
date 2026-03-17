import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
index = faiss.read_index("indexes/vector_index.faiss")

with open("indexes/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

data = json.load(open("training/msmarco_train.json"))

positives = [d for d in data if d["label"] == 1]

hard_negative_data = []

for item in tqdm(positives[:15000]):
    query = item["query"]
    positive_passage = item["passage"]

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, 20)

    hard_negatives = []
    for i in indices[0]:
        candidate = metadata[i]["text"]
        if candidate != positive_passage and len(candidate) > 50:
            hard_negatives.append(candidate)
        if len(hard_negatives) == 3:
            break

    hard_negative_data.append({
        "query": query,
        "passage": positive_passage,
        "label": 1.0
    })

    for hn in hard_negatives:
        hard_negative_data.append({
            "query": query,
            "passage": hn,
            "label": 0.0
        })

json.dump(hard_negative_data, open("training/msmarco_hard_negatives.json", "w"))
print(f"Total samples: {len(hard_negative_data)}")
print(f"Positives: {len(positives[:15000])}")
print(f"Hard negatives: {len(hard_negative_data) - len(positives[:15000])}")