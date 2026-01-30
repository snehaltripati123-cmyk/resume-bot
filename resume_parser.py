import os
import pdfplumber
from docx import Document
import pytesseract
from pdf2image import convert_from_path

# Set Tesseract path if on Windows (Adjust if your path is different)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(file_path):
    """
    Smart extraction: Tries FAST method first. Uses SLOW OCR only if needed.
    """
    ext = file_path.rsplit('.', 1)[-1].lower()
    text = ""

    try:
        # 1. Handle PDF
        if ext == 'pdf':
            # FAST WAY: Try to read text directly
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            except Exception as e:
                print(f"Fast PDF read failed: {e}")

            # SLOW WAY: If text is still empty (scanned PDF), use OCR
            if not text.strip():
                print(f"⚠️ Fast read failed for {file_path}. Switching to OCR (Slow)...")
                try:
                    images = convert_from_path(file_path)
                    for img in images:
                        text += pytesseract.image_to_string(img) + "\n"
                except Exception as e:
                    print(f"OCR failed: {e}")

        # 2. Handle DOCX (Fast)
        elif ext == 'docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        # 3. Handle Text Files
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return text.strip()