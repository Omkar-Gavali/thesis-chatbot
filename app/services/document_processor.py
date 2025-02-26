import fitz
from app.core.config import settings
from app.core.logging import logger

def process_pdf():
    logger.info(f"Processing PDF: {settings.thesis_pdf_path}")
    doc = fitz.open(settings.thesis_pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text):
    logger.info(f"Chunking text with size {settings.chunk_size} and overlap {settings.chunk_overlap}")
    chunks = []
    start = 0
    while start < len(text):
        end = start + settings.chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - settings.chunk_overlap
    return chunks
