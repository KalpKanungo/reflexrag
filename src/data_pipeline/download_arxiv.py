import os
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time

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
    "faster rcnn object detection",
    "retrieval augmented generation",
    "knowledge distillation",
    "contrastive learning",
    "self supervised learning",
    "transformer natural language processing",
    "object detection deep learning",
    "image segmentation deep learning",
    "neural machine translation",
    "question answering transformers",
    "graph neural networks",
    "reinforcement learning deep",
    "zero shot learning",
    "few shot learning",
    "multimodal learning",
    "semantic segmentation",
    "point cloud deep learning",
    "depth estimation monocular",
    "face recognition deep learning",
    "neural architecture search",
    "federated learning",
    "adversarial examples robustness",
    "medical image segmentation",
    "video understanding temporal",
    "optical flow estimation",
    "pose estimation human",
    "scene understanding indoor",
    "visual question answering",
    "image captioning attention",
    "text detection scene",
    "domain adaptation transfer learning",
    "panoptic segmentation",
    "3d object detection lidar",
    "action recognition video",
    "image generation synthesis",
    "cross modal retrieval",
    "document understanding layout",
    "anomaly detection unsupervised",
    "efficient neural networks mobile",
]

OUTPUT_DIR = "data/demo_corpus"

def search_arxiv(query, max_results=10):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    try:
        response = requests.get(ARXIV_API, params=params, timeout=30)
        root = ET.fromstring(response.text)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        pdf_links = []
        for entry in entries:
            links = entry.findall("{http://www.w3.org/2005/Atom}link")
            for link in links:
                if link.attrib.get("title") == "pdf":
                    pdf_links.append(link.attrib["href"])
        return pdf_links
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []

def download_pdf(url, save_path):
    try:
        r = requests.get(url, timeout=30)
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for topic in TOPICS:
        print(f"\nSearching: {topic}")
        pdfs = search_arxiv(topic, max_results=10)
        time.sleep(1)

        for url in tqdm(pdfs, desc=topic):
            name = url.split("/")[-1] + ".pdf"
            path = os.path.join(OUTPUT_DIR, name)

            if not os.path.exists(path):
                success = download_pdf(url, path)
                if success:
                    total_downloaded += 1
                else:
                    total_failed += 1
            else:
                total_skipped += 1

    print(f"\nDone!")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped (already exist): {total_skipped}")
    print(f"Failed: {total_failed}")
    print(f"Total papers in corpus: {len(os.listdir(OUTPUT_DIR))}")

if __name__ == "__main__":
    main()