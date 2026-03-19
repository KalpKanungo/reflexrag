from groq import Groq
import os 
class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def build_context(self, chunks):
        context = ""
        sources = []

        for i, c in enumerate(chunks):
            text = c["text"]

            if len(text.split()) < 40:
                continue

            if any(x in text.lower() for x in ["figure", "table", "http", "arxiv"]):
                continue

            context += f"[{i+1}] {text}\n"
            sources.append((i+1, text))

        return context.strip(), sources

    def generate(self, query, chunks):
        context, sources = self.build_context(chunks)

        prompt = f"""
                    You are a research assistant.

                    Answer the question using the provided context.

                    Rules:
                    - Use the context as the primary source
                    - Give a clear and direct answer first
                    - Then support it with information from the context
                    - Avoid phrases like "it seems", "it appears", "not specified"
                    - If some details are missing, add minimal general knowledge to complete the explanation
                    - Do NOT hallucinate specific claims not supported by context
                    - Keep the answer concise and confident (4-6 sentences)
                    - Use citations like [1], [2] where relevant

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    """

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return answer, sources