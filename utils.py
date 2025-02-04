import fitz
from PyPDF2 import PdfReader

def extract_pdf(file):
    reader = PdfReader(file)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    if 'incorrect startxref pointer(1)' in text:
        text = 'Error reading PDF - try different file'
    return text