import numpy as np
import faiss
import pickle

EMBEDDINGS_FILE = "indexes/embeddings.npy"
INDEX_FILE = "indexes/vector_index.faiss"

def main():
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

if __name__ == "__main__":
    main()