---
title: ReflexRAG
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🚀 ReflexRAG: Self-Correcting RAG System for Research Paper Analysis

> 🔍 A Retrieval-Augmented Generation (RAG) system for scientific literature with self-correction, grounding verification, and automated literature survey generation.

---

## 🤗 Live Demo

👉 **Try it here:**  
🔗 https://huggingface.co/spaces/KalpKanungo/Reflexrag  

---

## 📌 Overview

ReflexRAG is an end-to-end system designed to improve **scientific question answering** and **research paper analysis** using advanced RAG techniques.

It combines:
- Dense retrieval + reranking  
- Multi-hop reasoning  
- Hallucination detection & correction  
- Structured literature extraction  

---

## ⚙️ Key Features

### 🔎 1. Global Paper QA
- Query across a corpus of research papers  
- Uses multi-hop retrieval + reranking  
- Returns grounded answers with sources  

---

### 📄 2. Upload Paper QA
- Upload any research paper (PDF)  
- Ask questions directly on the document  
- Uses semantic chunking + embedding search  

---

### 📊 3. Literature Survey Generator
- Upload multiple papers  
- Automatically extracts:
  - Method  
  - Dataset  
  - Results  
  - Research gaps  
- Outputs a structured comparison table + CSV download  

---

### 🧠 4. Self-Correction Pipeline
- Detects hallucinations using grounding checks  
- Re-runs retrieval if answer is unreliable  
- Improves factual consistency  

---


---

## 🛠️ Tech Stack

- **Retrieval:** FAISS, Sentence Transformers (BGE)
- **Reranking:** Fine-tuned BGE Cross-Encoder (MS MARCO)
- **LLM:** Groq API (LLaMA-based models)
- **Frontend:** Gradio
- **PDF Processing:** PyMuPDF
- **Deployment:** HuggingFace Spaces

---



