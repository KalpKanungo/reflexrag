from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

class Reranker:
    def __init__(self, model_path, temperature=3.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 1
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            ignore_mismatched_sizes=True
        )
        self.model.to("cpu")
        self.model.eval()
        self.temperature = temperature

    def rerank(self, query, documents, top_k=5):
        pairs = [(query, doc["text"]) for doc in documents]
        inputs = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scaled = (logits - logits.mean()) / (logits.std() + 1e-8)
            scores = torch.sigmoid(scaled / self.temperature).tolist()

        if isinstance(scores, float):
            scores = [scores]

        scored_docs = []
        for i, score in enumerate(scores):
            doc = dict(documents[i])
            doc["score"] = score
            scored_docs.append(doc)

        ranked = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]