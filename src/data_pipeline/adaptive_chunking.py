import json
import os

INPUT_FILE = "data/processed_chunks/papers.json"
OUTPUT_FILE = "data/processed_chunks/chunks.json"

CHUNK_SIZE = 150

def chunk_text(text):
    text = text.replace("\n"," ")
    text = " ".join(text.split())
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = words[i:i+CHUNK_SIZE]
        if len(chunk) > 50:
            chunks.append(" ".join(chunk))
    return chunks

def main():
    with open(INPUT_FILE) as f:
        papers = json.load(f)

    all_chunks = []

    for paper in papers:
        paper_id = paper["paper_id"]
        text = paper["text"]

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "paper_id": paper_id,
                "chunk_id": idx,
                "text": chunk
            })

    with open(OUTPUT_FILE,"w") as f:
        json.dump(all_chunks,f)

if __name__ == "__main__":
    main()