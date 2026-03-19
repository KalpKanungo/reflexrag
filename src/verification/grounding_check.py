from sentence_transformers import SentenceTransformer
import numpy as np

class GroundingChecker:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def compute_similarity(self, answer, chunks):
        answer_emb = self.model.encode([answer], normalize_embeddings=True)

        chunk_texts = [c["text"] for c in chunks]
        chunk_embs = self.model.encode(chunk_texts, normalize_embeddings=True)

        sims = np.dot(chunk_embs, answer_emb.T).squeeze()

        return float(np.max(sims))

    def is_grounded(self, answer, chunks, threshold=0.6):
        score = self.compute_similarity(answer, chunks)
        return score >= threshold, score