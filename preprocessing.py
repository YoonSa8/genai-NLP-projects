import fitz  # PyMuPDF
from docx import Document
import os
import re


def extract_resume_text(file_path):
    #Extract raw text from a resume file (PDF or DOCX)
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == '.pdf':
        return _extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return _extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_clean_resume_text(file_path):
    #Extract and clean resume text for embedding models.
    raw_text = extract_resume_text(file_path)
    return clean_text(raw_text)


def _extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()


def _extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()


def clean_text(text):
    #Basic preprocessing: lowercasing, removing punctuation, extra spaces, etc.
    text = text.lower()
    text = re.sub(r"\s+", " ", text)                      # Collapse whitespace
    text = re.sub(r"[^\w\s]", "", text)                   # Remove punctuation
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", text)  # Remove dates (optional)
    return text.strip()

