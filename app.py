import gradio as gr
import multiprocessing
import torch
import tempfile

multiprocessing.set_start_method("spawn", force=True)
torch.set_num_threads(1)

from src.reasoning.multi_hop_pipeline import MultiHopRetriever
from src.reasoning.answer_generation import AnswerGenerator
from src.verification.grounding_check import GroundingChecker
from src.verification.self_correction import SelfCorrector
from src.data_pipeline.user_pdf_pipeline import extract_text_from_pdf
from src.literature.survey_generator import SurveyGenerator

retriever = MultiHopRetriever("kalpkanungo/reranker_finetuned_v2")
generator = AnswerGenerator()
checker = GroundingChecker()
corrector = SelfCorrector()
survey_generator = SurveyGenerator()

def answer_query(query):
    if not query.strip():
        return "Please enter a question.", "", "", ""

    answer, sources, score, corrected = corrector.correct(
        query,
        retriever,
        generator,
        checker
    )

    sources_text = ""
    for i, s in sources:
        sources_text += f"[{i}] {s[:200]}...\n\n"

    return answer, f"{score:.3f}", str(corrected), sources_text


def answer_uploaded(file, query):
    if file is None or not query.strip():
        return "Upload a file and enter a question."

    text = extract_text_from_pdf(file)

    import re
    sections = re.split(r'\n(?=[A-Z][A-Z\s]{3,})', text)

    chunks = []
    for sec in sections:
        if len(sec.strip()) > 200:
            for i in range(0, len(sec), 800):
                chunks.append({"text": sec[i:i+800]})

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    query_emb = model.encode([query], normalize_embeddings=True)
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = model.encode(chunk_texts, normalize_embeddings=True)

    scores = np.dot(chunk_embs, query_emb.T).squeeze()

    top_k = 5
    top_indices = np.argsort(scores)[-top_k:][::-1]

    selected_chunks = [chunks[i] for i in top_indices]

    answer, _ = generator.generate(query, selected_chunks)

    return answer


def generate_survey(files):
    import pandas as pd

    if not files:
        return pd.DataFrame(), None

    df = survey_generator.process_papers(files)

    if df is None or df.empty:
        return pd.DataFrame(), None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)

    return df, temp_file.name


with gr.Blocks(title="ReflexRAG") as app:

    gr.Markdown("# 🔍 ReflexRAG")
    gr.Markdown("Self-Correcting Research Paper Analysis System")

    with gr.Tab("🌐 Global Paper QA"):
        query = gr.Textbox(label="Enter your question")
        btn = gr.Button("Get Answer")

        answer = gr.Textbox(label="Answer", lines=6)
        score = gr.Textbox(label="Grounding Score")
        corrected = gr.Textbox(label="Self-Corrected")
        sources = gr.Textbox(label="Sources", lines=10)

        btn.click(answer_query, inputs=query, outputs=[answer, score, corrected, sources])

    with gr.Tab("📄 Upload Paper QA"):
        file = gr.File(label="Upload PDF")
        query2 = gr.Textbox(label="Ask a question about the uploaded paper")
        btn2 = gr.Button("Get Answer")

        answer2 = gr.Textbox(label="Answer", lines=6)

        btn2.click(answer_uploaded, inputs=[file, query2], outputs=answer2)

    with gr.Tab("📊 Literature Survey"):
        files = gr.File(file_count="multiple", label="Upload multiple research papers")
        btn3 = gr.Button("Generate Survey")

        table = gr.Dataframe()
        download = gr.File(label="Download CSV")

        btn3.click(generate_survey, inputs=files, outputs=[table, download])

app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)