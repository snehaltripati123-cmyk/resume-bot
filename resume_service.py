from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
import uuid

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def extract_text(file_path):
    elements = partition(filename=file_path)
    return "\n".join([str(e) for e in elements])

def chunk_text(text, size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)
