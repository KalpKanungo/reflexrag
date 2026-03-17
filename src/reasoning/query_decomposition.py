from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QueryDecomposer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.model.eval()

    def decompose(self, query):
        sub_queries = []
        
        prompts = [
            f"What is the definition and architecture of the main concept in: {query}",
            f"What are the key advantages and improvements of: {query}",
            f"What experiments or benchmarks compare the methods in: {query}",
        ]
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=2,
                    early_stopping=True
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if len(text) > 5 and text.lower() != query.lower():
                sub_queries.append(text)

        if len(sub_queries) == 0:
            return self._rule_based_decompose(query)

        return sub_queries[:3]

    def _rule_based_decompose(self, query):
        stop_words = {"is", "are", "was", "were", "the", "a", "an", "why",
                      "how", "what", "which", "does", "do", "than"}

        words = [w for w in query.lower().split() if w not in stop_words]

        sub_queries = [
            query,
            " ".join(words[:len(words)//2 + 1]),
            " ".join(words[len(words)//2:]),
        ]

        return [q for q in sub_queries if len(q) > 5][:3]