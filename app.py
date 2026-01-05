from flask import Flask, request, jsonify, render_template
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import Resume
from resume_parser import extract_text
from groq_service import answer_question, generate_chat_title
import uuid
import os

# Create Tables
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Handle single or multiple files
    files = request.files.getlist('files[]')
    if not files:
        file = request.files.get('file')
        if file: files = [file]

    if not files: return jsonify({"error": "No files"}), 400

    db = SessionLocal()
    saved_ids = []
    first_title = ""

    for file in files:
        if file.filename == '': continue
        if not first_title: first_title = file.filename

        file_id = uuid.uuid4()
        path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")
        file.save(path)

        # 1. Fast Extraction
        text = extract_text(path)
        if not text: text = ""

        # 2. Save Immediately (No embedding, no delay)
        resume = Resume(
            id=file_id, 
            filename=file.filename, 
            raw_text=text, 
            analysis={"title": file.filename} # Default title
        )
        db.add(resume)
        db.commit()
        saved_ids.append(str(file_id))

    db.close()
    # Returns the exact JSON structure your frontend expects
    return jsonify({"resume_id": saved_ids[0], "title": first_title})

@app.route("/history", methods=["GET"])
def get_history():
    db = SessionLocal()
    resumes = db.query(Resume).order_by(Resume.uploaded_at.desc()).all()
    history = []
    for r in resumes:
        title = r.analysis.get("title", r.filename) if r.analysis else r.filename
        history.append({"id": str(r.id), "title": title})
    db.close()
    return jsonify(history)

# --- THIS IS THE FIX: RENAMED FROM '/chat' TO '/ask' ---
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    resume_id = data.get("resume_id")
    question = data.get("question")
    
    db = SessionLocal()
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume: 
        db.close()
        return jsonify({"answer": "Resume not found."})

    # Direct Chat
    answer = answer_question(resume.raw_text, question)
    
    # Auto-Rename
    new_title = None
    try:
        if resume.analysis and resume.analysis.get("title") == resume.filename:
            new_title = generate_chat_title(question)
            resume.analysis["title"] = new_title
            resume.analysis = dict(resume.analysis)
            db.commit()
    except: pass

    db.close()
    return jsonify({"answer": answer, "new_title": new_title})

if __name__ == "__main__":
    app.run(debug=True, port=5000)