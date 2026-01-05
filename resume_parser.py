import os
from pypdf import PdfReader
import docx  # from python-docx

# OCR Support (Optional - prevents crash if not installed)
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

def extract_text(file_path):
    """
    Main function to detect file type and call the right extractor.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return extract_from_docx(file_path)
    elif ext == ".txt":
        return extract_from_txt(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_from_image(file_path)
    else:
        return f"Unsupported file format: {ext}"

# --- Helper Functions ---

def extract_from_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading TXT: {e}"

def extract_from_image(path):
    if not HAS_OCR:
        return "[Error] Image detected but Tesseract-OCR is not installed."
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        return f"Error reading Image: {e}"