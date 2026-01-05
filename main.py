from fastapi import FastAPI, UploadFile, File
from database import engine, SessionLocal
from models import Base, Resume, ResumeChunk
from resume_service import extract_text, chunk_text, embed_chunks
from search_service import search_chunks
from llm_service import answer_with_groq
from question_utils import normalize_question, detect_intent
import shutil
import uuid


app = FastAPI()

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"message": "Resume Bot API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    db = SessionLocal()
    resume = Resume(id=uuid.uuid4(), raw_text=text)
    db.add(resume)

    for c, e in zip(chunks, embeddings):
        db.add(ResumeChunk(
            id=uuid.uuid4(),
            resume_id=resume.id,
            content=c,
            embedding=e.tolist()
        ))

    resume_id = resume.id  # store before session close

    db.commit()
    db.close()

    return {"resume_id": str(resume_id), "status": "processed"}

@app.post("/resume/ask")
async def ask_resume(resume_id: str, question: str):
    norm_q = normalize_question(question)
    intent = detect_intent(norm_q)

    chunks = search_chunks(resume_id, norm_q)

    if not chunks:
        return {"answer": "This information is not present in the resume."}

    context = "\n\n".join([c.content for c in chunks[:5]])

    answer = answer_with_groq(norm_q, context, intent)

    return {"answer": answer}

