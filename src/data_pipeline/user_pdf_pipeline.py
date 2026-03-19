import fitz

def extract_text_from_pdf(file):
    doc = fitz.open(file.name)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text