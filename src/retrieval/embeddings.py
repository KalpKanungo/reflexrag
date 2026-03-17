import json
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "data/processed_chunks/chunks.json"
EMBEDDINGS_FILE = "indexes/embeddings.npy"
METADATA_FILE = "indexes/metadata.pkl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def main():
    os.makedirs("indexes", exist_ok=True)
    chunks = json.load(open(INPUT_FILE))
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Total chunks embedded: {len(chunks)}")

if __name__ == "__main__":
    main()