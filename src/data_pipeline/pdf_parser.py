import os
import json
import fitz
from tqdm import tqdm

INPUT_DIR = "data/demo_corpus"
OUTPUT_FILE = "data/processed_chunks/papers.json"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def main():
    papers = []
    for file in tqdm(os.listdir(INPUT_DIR)):
        if file.endswith(".pdf"):
            path = os.path.join(INPUT_DIR, file)
            text = extract_text(path)
            papers.append({
                "paper_id": file.replace(".pdf",""),
                "text": text
            })
    os.makedirs("data/processed_chunks", exist_ok=True)
    with open(OUTPUT_FILE,"w") as f:
        json.dump(papers,f)

if __name__ == "__main__":
    main()