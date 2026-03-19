from groq import Groq

class InfoExtractor:
    def __init__(self):
        self.client = Groq()

    def extract(self, text):
        prompt = f"""
                    Extract structured information.

                    Return ONLY JSON.

                    Fields:
                    - title (short)
                    - method (1 sentence)
                    - dataset (1 sentence)
                    - results (1 sentence)
                    - gap (1 sentence)

                    Rules:
                    - ALL fields MUST be filled
                    - If missing → infer from context
                    - Keep each field concise (max 20 words)

                    Format:
                    {{
                    "title": "...",
                    "method": "...",
                    "dataset": "...",
                    "results": "...",
                    "gap": "..."
                    }}

                    Text:
                    {text[:2500]}
                    """

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content