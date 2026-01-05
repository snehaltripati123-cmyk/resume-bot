from database import SessionLocal
from models import ResumeChunk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def search_chunks(resume_id, query, top_k=5):
    query_vec = model.encode([query])[0].tolist()

    db = SessionLocal()
    results = db.query(ResumeChunk).filter(
        ResumeChunk.resume_id == resume_id
    ).order_by(
        ResumeChunk.embedding.cosine_distance(query_vec)
    ).limit(top_k).all()

    db.close()
    return results
