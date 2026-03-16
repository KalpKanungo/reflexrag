import os
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

ARXIV_API = "http://export.arxiv.org/api/query"

TOPICS = [
    "vision transformer",
    "clip vision language",
    "detr object detection",
    "diffusion models",
    "attention is all you need",
    "bert language model",
    "gpt language model",
    "resnet deep residual learning",
    "alexnet imagenet classification",
    "generative adversarial networks",
    "yolo object detection",
    "faster rcnn object detection"
]

OUTPUT_DIR = "data/demo_corpus"

def search_arxiv(query, max_results=5):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(ARXIV_API, params=params)
    root = ET.fromstring(response.text)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    pdf_links = []
    for entry in entries:
        links = entry.findall("{http://www.w3.org/2005/Atom}link")
        for link in links:
            if link.attrib.get("title") == "pdf":
                pdf_links.append(link.attrib["href"])
    return pdf_links

def download_pdf(url, save_path):
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for topic in TOPICS:
        pdfs = search_arxiv(topic, max_results=3)
        for url in tqdm(pdfs):
            name = url.split("/")[-1] + ".pdf"
            path = os.path.join(OUTPUT_DIR, name)
            if not os.path.exists(path):
                download_pdf(url, path)

if __name__ == "__main__":
    main()