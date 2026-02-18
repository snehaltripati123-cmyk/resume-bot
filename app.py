# import os, json, uuid
# from flask import Flask, request, jsonify, render_template
# from PyPDF2 import PdfReader
# from groq import Groq
# from dotenv import load_dotenv

# # 1. Load Environment Variables
# load_dotenv()

# app = Flask(__name__)

# # 2. Configuration
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
# PROMPT_FILE = "prompt.txt"

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# client = Groq(api_key=GROQ_API_KEY)

# # --- GLOBAL STORAGE ---
# SESSIONS = {} 
# CURRENT_RESUME = {"text": "", "filename": ""}

# # --- HELPER FUNCTIONS ---
# def get_system_prompt():
#     if os.path.exists(PROMPT_FILE):
#         with open(PROMPT_FILE, "r") as f:
#             return f.read()
#     return "You are a helpful HR assistant."

# def extract_text_from_file(filepath):
#     try:
#         with open(filepath, 'rb') as f:
#             reader = PdfReader(f)
#             text = "\n".join(p.extract_text() or "" for p in reader.pages)
#         return text
#     except Exception as e:
#         print(f"❌ Error reading PDF: {e}")
#         return ""

# def query_groq(messages, json_mode=False):
#     kwargs = {
#         "model": "llama-3.3-70b-versatile",
#         "messages": messages,
#         "temperature": 0.3,
#         "max_tokens": 2048
#     }
#     if json_mode:
#         kwargs["response_format"] = {"type": "json_object"}
#     try:
#         completion = client.chat.completions.create(**kwargs)
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error: {str(e)}"

# # --- ROUTES ---

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/comparison")
# def comparison_page():
#     return render_template("comparison.html")

# # ==========================================
# # 1. SINGLE RESUME SYSTEM (DEBUGGED)
# # ==========================================

# @app.route("/upload", methods=["POST"])
# def upload_resume():
#     print(f"📥 Receiving upload request... Files: {request.files}") # Debug Print

#     if "resume" not in request.files:
#         print("❌ Error: 'resume' key missing in request.files")
#         return jsonify({"error": "No resume uploaded. Check JS FormData name."}), 400

#     file = request.files["resume"]
#     if file.filename == "":
#         print("❌ Error: Empty filename")
#         return jsonify({"error": "Empty filename"}), 400

#     try:
#         # Save to disk
#         filename = f"{uuid.uuid4().hex}_{file.filename}"
#         filepath = os.path.join(UPLOAD_DIR, filename)
#         file.save(filepath)
#         print(f"✅ File saved to: {filepath}")

#         # Extract text
#         text = extract_text_from_file(filepath)
#         if not text:
#             print("❌ Error: Extracted text is empty")
#             return jsonify({"error": "Could not read PDF text"}), 400

#         # Update State
#         CURRENT_RESUME["text"] = text
#         CURRENT_RESUME["filename"] = file.filename
#         print(f"✅ Text extracted ({len(text)} chars). Ready to chat.")

#         return jsonify({
#             "message": "File uploaded successfully",
#             "filename": file.filename 
#         })
#     except Exception as e:
#         print(f"❌ Exception: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route("/ask", methods=["POST"])
# def ask_single():
#     # ... (existing code to get question and context) ...
#     answer_raw = query_groq(messages, json_mode=True) # Ensure JSON mode is on
    
#     try:
#         data = json.loads(answer_raw)
#         # Only send the text string back to the frontend
#         return jsonify({"answer": data.get("response", answer_raw)})
#     except:
#         return jsonify({"answer": answer_raw})


# # ==========================================
# # 2. MULTI-RESUME SYSTEM (UNCHANGED)
# # ==========================================

# @app.route("/comparison/create", methods=["POST"])
# def create_comparison():
#     cid = str(uuid.uuid4())
#     SESSIONS[cid] = []
#     return jsonify({"comparison_id": cid})

# @app.route("/comparison/<cid>/upload", methods=["POST"])
# def upload_candidates(cid):
#     files = request.files.getlist("files[]")
#     if cid not in SESSIONS: SESSIONS[cid] = [] 

#     count = 0
#     for file in files:
#         if file.filename == '': continue
#         filename = f"{cid}_{file.filename}"
#         filepath = os.path.join(UPLOAD_DIR, filename)
#         file.save(filepath)
        
#         text = extract_text_from_file(filepath)
#         if text:
#             SESSIONS[cid].append({"filename": file.filename, "text": text})
#             count += 1
#     return jsonify({"status": "ok", "count": count})

# @app.route("/comparison/<cid>/candidates")
# def list_candidates(cid):
#     if cid not in SESSIONS: return jsonify([])
#     return jsonify([{"filename": c["filename"], "status": "Ready"} for c in SESSIONS[cid]])

# @app.route("/comparison/<cid>/chat", methods=["POST"])
# def multi_resume_chat(cid):
#     if cid not in SESSIONS or not SESSIONS[cid]:
#         return jsonify({"response": "Please upload resumes first.", "graph_data": None})
    
#     data = request.json
#     sys_prompt = get_system_prompt()
#     resumes = SESSIONS[cid]
#     context = "\n".join([f"Candidate: {r['filename']}\nText: {r['text'][:3000]}..." for r in resumes])

#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": f"Resumes:\n{context}\n\nQuery: {data.get('message', '')}"}
#     ]

#     response_raw = query_groq(messages, json_mode=True)
#     try:
#         return jsonify(json.loads(response_raw))
#     except:
#         return jsonify({"response": response_raw, "graph_data": None})

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


# import os, json, uuid
# from flask import Flask, request, jsonify, render_template
# from PyPDF2 import PdfReader
# from groq import Groq
# from dotenv import load_dotenv

# # Load Environment Variables
# load_dotenv()
# app = Flask(__name__)

# # Configuration
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# UPLOAD_DIR = "uploads"
# PROMPT_FILE = "prompt.txt"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # --- GLOBAL STORAGE ---
# SESSIONS = {} 
# # FIXED: Re-added missing dictionary for single resume mode
# CURRENT_RESUME = {"text": "", "filename": ""}

# # --- HELPER FUNCTIONS ---

# def get_system_prompt():
#     """Reads AI instructions from an external file."""
#     if os.path.exists(PROMPT_FILE):
#         with open(PROMPT_FILE, "r") as f:
#             return f.read()
#     return "You are a helpful HR assistant."

# def extract_text_from_file(filepath):
#     """Safely extracts text from a saved PDF file to avoid EOF crashes."""
#     try:
#         with open(filepath, 'rb') as f:
#             reader = PdfReader(f)
#             return "\n".join(p.extract_text() or "" for p in reader.pages)
#     except Exception as e:
#         print(f"Error reading PDF: {e}")
#         return ""

# def query_groq(messages, json_mode=False):
#     """Standardized Groq API call."""
#     kwargs = {
#         "model": "llama-3.3-70b-versatile",
#         "messages": messages,
#         "temperature": 0.3,
#         "max_tokens": 2048
#     }
#     if json_mode:
#         kwargs["response_format"] = {"type": "json_object"}
    
#     try:
#         completion = client.chat.completions.create(**kwargs)
#         return completion.choices[0].message.content
#     except Exception as e:
#         return json.dumps({"response": f"Error: {str(e)}", "graph_data": None})

# # --- SINGLE RESUME ROUTES ---

# @app.route("/")
# def index(): 
#     return render_template("index.html")

# @app.route("/upload", methods=["POST"])
# def upload_single_resume():
#     """Fixed upload for single resume mode."""
#     if "resume" not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files["resume"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     # Save file to disk to prevent 'EOF marker' crashes
#     filename = f"single_{uuid.uuid4().hex}_{file.filename}"
#     filepath = os.path.join(UPLOAD_DIR, filename)
#     file.save(filepath)

#     text = extract_text_from_file(filepath)
#     if not text:
#         return jsonify({"error": "Failed to extract text from PDF"}), 400

#     # Store in the global CURRENT_RESUME dictionary
#     CURRENT_RESUME["text"] = text
#     CURRENT_RESUME["filename"] = file.filename

#     return jsonify({"message": "success", "filename": file.filename})

# @app.route("/ask", methods=["POST"])
# def ask_single():
#     """Fixed chat route for single resume."""
#     data = request.json
#     question = data.get("question")
    
#     if not CURRENT_RESUME["text"]:
#         return jsonify({"answer": "Please upload a resume first."})

#     messages = [
#         {"role": "system", "content": get_system_prompt()},
#         {"role": "user", "content": f"Resume Content:\n{CURRENT_RESUME['text'][:10000]}\n\nQuestion: {question}"}
#     ]
    
#     # Force JSON mode to parse out only the polished text
#     answer_raw = query_groq(messages, json_mode=True)
#     try:
#         parsed = json.loads(answer_raw)
#         # Returns only the 'response' string (no brackets)
#         return jsonify({"answer": parsed.get("response", answer_raw)})
#     except:
#         return jsonify({"answer": answer_raw})

# # --- MULTI-RESUME COMPARISON ROUTES ---

# @app.route("/comparison")
# def comparison_page(): 
#     return render_template("comparison.html")

# @app.route("/comparison/create", methods=["POST"])
# def create_comparison():
#     cid = str(uuid.uuid4())
#     SESSIONS[cid] = []
#     return jsonify({"comparison_id": cid})

# @app.route("/comparison/<cid>/upload", methods=["POST"])
# def upload_candidates(cid):
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No file part"}), 400
        
#     files = request.files.getlist("files[]")
#     if cid not in SESSIONS:
#         SESSIONS[cid] = []

#     count = 0
#     for f in files:
#         if f.filename == '': continue
#         path = os.path.join(UPLOAD_DIR, f"{cid}_{f.filename}")
#         f.save(path)
        
#         text = extract_text_from_file(path)
#         if text:
#             SESSIONS[cid].append({"filename": f.filename, "text": text})
#             count += 1
        
#     return jsonify({"status": "ok", "count": count})

# @app.route("/comparison/<cid>/chat", methods=["POST"])
# def multi_resume_chat(cid):
#     data = request.json
#     resumes = SESSIONS.get(cid, [])
    
#     if not resumes:
#         return jsonify({"response": "Please upload resumes first.", "graph_data": None})

#     context = "\n".join([f"Candidate: {r['filename']}\nText: {r['text'][:2000]}" for r in resumes])
#     messages = [
#         {"role": "system", "content": get_system_prompt()},
#         {"role": "user", "content": f"Context: {context}\nQuery: {data['message']}"}
#     ]

#     raw_ai_output = query_groq(messages, json_mode=True)
#     try:
#         parsed = json.loads(raw_ai_output)
#         # Only return graph_data if the AI actually generated NEW data for THIS query
#         return jsonify({
#             "response": parsed.get("response", "I've updated the analysis for you."), 
#             "graph_data": parsed.get("graph_data", None)
#         })
#     except:
#         return jsonify({"response": "I had trouble generating a new chart for that question.", "graph_data": None})

# @app.route("/comparison/<cid>/suggestions", methods=["GET"])
# def get_smart_suggestions(cid):
#     resumes = SESSIONS.get(cid, [])
#     if not resumes:
#         return jsonify({"suggestions": []})

#     summary = "\n".join([f"Candidate: {r['filename']}, Content: {r['text'][:1000]}" for r in resumes])
#     prompt = [
#         {"role": "system", "content": "You are a recruitment assistant. Based on these resumes, suggest 4 short, highly specific comparison questions. Return ONLY JSON: {\"suggestions\": [\"question1\", \"question2\", ...]}"},
#         {"role": "user", "content": summary}
#     ]
    
#     try:
#         raw_response = query_groq(prompt, json_mode=True)
#         return jsonify(json.loads(raw_response))
#     except Exception:
#         return jsonify({"suggestions": ["Compare experience", "Top skills", "Education", "Best fit"]})

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)



# import os, json, uuid
# from flask import Flask, request, jsonify, render_template
# from PyPDF2 import PdfReader
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()
# app = Flask(__name__)

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# UPLOAD_DIR = "uploads"
# PROMPT_FILE = "prompt.txt"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# SESSIONS = {} 
# CURRENT_RESUME = {"text": "", "filename": ""} # Global storage for single mode

# def get_system_prompt():
#     if os.path.exists(PROMPT_FILE):
#         with open(PROMPT_FILE, "r") as f:
#             return f.read()
#     # Fallback must include 'json' to prevent API error 400
#     return "You are an HR assistant. Return your answer in json format with 'response' and 'graph_data' keys."

# def extract_text_from_file(filepath):
#     try:
#         with open(filepath, 'rb') as f:
#             reader = PdfReader(f)
#             return "\n".join(p.extract_text() or "" for p in reader.pages)
#     except Exception as e:
#         print(f"Error reading PDF: {e}")
#         return ""

# def query_groq(messages, json_mode=False):
#     """Standardized Groq API call using the updated instant model."""
#     kwargs = {
#         "model": "llama-3.1-8b-instant",  # Updated model string
#         "messages": messages,
#         "temperature": 0.3,
#         "max_tokens": 2048
#     }
#     if json_mode:
#         kwargs["response_format"] = {"type": "json_object"}
    
#     try:
#         completion = client.chat.completions.create(**kwargs)
#         return completion.choices[0].message.content
#     except Exception as e:
#         # Fallback to prevent the 'Unexpected token <' frontend error
#         return json.dumps({"response": f"API Error: {str(e)}", "graph_data": None})
# # --- ROUTES ---

# @app.route("/")
# def index(): return render_template("index.html")

# @app.route("/upload", methods=["POST"])
# def upload_single_resume():
#     if "resume" not in request.files: return jsonify({"error": "No file"}), 400
#     file = request.files["resume"]
#     filename = f"single_{uuid.uuid4().hex}_{file.filename}"
#     filepath = os.path.join(UPLOAD_DIR, filename)
#     file.save(filepath)
#     text = extract_text_from_file(filepath)
#     if not text: return jsonify({"error": "Extract failed"}), 400
#     CURRENT_RESUME["text"] = text
#     CURRENT_RESUME["filename"] = file.filename
#     return jsonify({"message": "success", "filename": file.filename})

# @app.route("/ask", methods=["POST"])
# def ask_single():
#     data = request.json
#     if not CURRENT_RESUME["text"]: return jsonify({"answer": "Upload first."})
#     messages = [
#         {"role": "system", "content": get_system_prompt()},
#         {"role": "user", "content": f"Resume: {CURRENT_RESUME['text'][:5000]}\nQ: {data.get('question')}"}
#     ]
#     raw = query_groq(messages, json_mode=True)
#     try:
#         parsed = json.loads(raw)
#         return jsonify({"answer": parsed.get("response", raw)})
#     except: return jsonify({"answer": raw})

# @app.route("/comparison")
# def comparison_page(): return render_template("comparison.html")

# @app.route("/comparison/<cid>/upload", methods=["POST"])
# def upload_candidates(cid):
#     files = request.files.getlist("files[]")
    
#     # Ensure SESSIONS[cid] is a dictionary with the correct structure
#     if cid not in SESSIONS:
#         SESSIONS[cid] = {"resumes": [], "history": []}

#     count = 0
#     for f in files:
#         if f.filename == '': continue
#         path = os.path.join(UPLOAD_DIR, f"{cid}_{f.filename}")
#         f.save(path)
#         text = extract_text_from_file(path)
#         if text:
#             # FIX: Access the "resumes" key before appending
#             SESSIONS[cid]["resumes"].append({"filename": f.filename, "text": text})
#             count += 1
            
#     return jsonify({"status": "ok", "count": count})
# @app.route("/comparison/<cid>/chat", methods=["POST"])
# def multi_resume_chat(cid):
#     data = request.json
#     resumes = SESSIONS.get(cid, [])
    
#     if not resumes:
#         return jsonify({"response": "Please upload resumes first.", "graph_data": None})

#     # 1. Initialize history for this session if it doesn't exist
#     if "history" not in SESSIONS[cid]:
#         SESSIONS[cid]["history"] = []

#     # 2. Build context from resumes
#     resume_context = "\n".join([f"Candidate: {r['filename']}\nText: {r['text'][:1500]}" for r in resumes])
    
#     # 3. Create the prompt with history
#     # We include the resume context in the system prompt so it's always "remembered"
#     messages = [
#         {"role": "system", "content": f"{get_system_prompt()}\n\nRECRUITMENT DATA:\n{resume_context}"}
#     ]
    
#     # 4. Append previous conversation rounds to the messages
#     # This allows follow-ups like "What about their education?" after asking about skills
#     messages.extend(SESSIONS[cid]["history"])
    
#     # 5. Add the current user question
#     messages.append({"role": "user", "content": data['message']})

#     raw_ai_output = query_groq(messages, json_mode=True)
    
#     try:
#         parsed = json.loads(raw_ai_output)
#         ai_response = parsed.get("response", "")
        
#         # 6. Save this round to history (limit to last 10 messages to save tokens)
#         SESSIONS[cid]["history"].append({"role": "user", "content": data['message']})
#         SESSIONS[cid]["history"].append({"role": "assistant", "content": ai_response})
#         SESSIONS[cid]["history"] = SESSIONS[cid]["history"][-10:] 

#         return jsonify({
#             "response": ai_response, 
#             "graph_data": parsed.get("graph_data", None)
#         })
#     except:
#         return jsonify({"response": "I encountered an error. Could you rephrase that?", "graph_data": None})
# @app.route("/comparison/create", methods=["POST"])
# def create_comparison():
#     cid = str(uuid.uuid4())
#     # FIX: Initialize as a dictionary, not a list
#     SESSIONS[cid] = {
#         "resumes": [],
#         "history": []
#     }
#     return jsonify({"comparison_id": cid})


# if __name__ == "__main__":
#     app.run(debug=True, port=5000)




import os, json, uuid
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv
from ats_engine import hybrid_ats_score


load_dotenv()
app = Flask(__name__)

# --- CONFIGURATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
UPLOAD_DIR = "uploads"
PROMPT_FILE = "prompt.txt"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global Storage
SESSIONS = {} 
CURRENT_RESUME = {"text": "", "filename": ""} # For Single Chat Mode

# --- HELPER FUNCTIONS ---

def get_system_prompt():
    """Reads AI instructions. Must mention 'json' for Groq JSON mode."""
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return f.read()
    return "You are a professional HR assistant. You must respond strictly in JSON format with 'response' and 'graph_data' keys."

def extract_text_from_file(filepath):
    """Safely extracts text from a saved PDF file."""
    try:
        with open(filepath, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def query_groq(messages, json_mode=False):
    """Standardized Groq API call using Llama-3.1-8b-instant for better rate limits."""
    kwargs = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    
    try:
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    except Exception as e:
        return json.dumps({"response": f"API Error: {str(e)}", "graph_data": None})

# --- ROUTES ---

@app.route("/")
def index(): 
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_single_resume():
    """Handles single resume upload for index.html."""
    if "resume" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["resume"]
    filename = f"single_{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    text = extract_text_from_file(filepath)
    if not text: return jsonify({"error": "Extract failed"}), 400
    CURRENT_RESUME["text"] = text
    CURRENT_RESUME["filename"] = file.filename
    return jsonify({"message": "success", "filename": file.filename})

@app.route("/ask", methods=["POST"])
def ask_single():
    """Handles questions for the single resume chatbot."""
    data = request.json
    if not CURRENT_RESUME["text"]: return jsonify({"answer": "Upload first."})
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": f"Resume Content:\n{CURRENT_RESUME['text'][:5000]}\n\nQuestion: {data.get('question')}"}
    ]
    raw = query_groq(messages, json_mode=True)
    try:
        parsed = json.loads(raw)
        return jsonify({"answer": parsed.get("response", raw)})
    except: return jsonify({"answer": raw})

@app.route("/comparison")
def comparison_page(): 
    return render_template("comparison.html")

@app.route("/comparison/create", methods=["POST"])
def create_comparison():
    """Initializes a new comparison session with dictionary structure."""
    cid = str(uuid.uuid4())
    SESSIONS[cid] = {
        "resumes": [],
        "history": []
    }
    return jsonify({"comparison_id": cid})

@app.route("/comparison/<cid>/upload", methods=["POST"])
def upload_candidates(cid):
    """Saves multiple resumes into the session resumes list."""
    if cid not in SESSIONS:
        SESSIONS[cid] = {"resumes": [], "history": []}
    
    files = request.files.getlist("files[]")
    count = 0
    for f in files:
        if f.filename == '': continue
        path = os.path.join(UPLOAD_DIR, f"{cid}_{f.filename}")
        f.save(path)
        text = extract_text_from_file(path)
        if text:
            # CORRECT: Append to the resumes list inside the dictionary
            SESSIONS[cid]["resumes"].append({"filename": f.filename, "text": text})
            count += 1
            
    return jsonify({"status": "ok", "count": count})

@app.route("/comparison/<cid>/chat", methods=["POST"])
def multi_resume_chat(cid):
    """Smart chat that handles follow-up questions using conversation history."""
    data = request.json
    session = SESSIONS.get(cid)
    
    if not session or not session["resumes"]:
        return jsonify({"response": "Please upload resumes first.", "graph_data": None})

    # 1. Initialize history if missing
    if "history" not in session:
        session["history"] = []

    # 2. Build context from resumes (Truncated to save tokens)
    resume_context = "\n".join([f"Candidate: {r['filename']}\nText: {r['text'][:1500]}" for r in session["resumes"]])
    
    # 3. Construct message list (System Prompt + History + New Question)
    messages = [
        {"role": "system", "content": f"{get_system_prompt()}\n\nAVAILABLE DATA:\n{resume_context}"}
    ]
    
    # Add previous conversation context
    messages.extend(session["history"])
    
    # Add current user query
    messages.append({"role": "user", "content": data['message']})

    raw_ai_output = query_groq(messages, json_mode=True)
    
    try:
        parsed = json.loads(raw_ai_output)
        ai_response = parsed.get("response", "Analysis complete.")
        
        # 4. Save to history (Keep last 10 messages for memory)
        session["history"].append({"role": "user", "content": data['message']})
        session["history"].append({"role": "assistant", "content": ai_response})
        session["history"] = session["history"][-10:] 

        return jsonify({
            "response": ai_response, 
            "graph_data": parsed.get("graph_data", None)
        })
    except:
        return jsonify({"response": "I'm sorry, I couldn't parse that analysis. Could you ask again?", "graph_data": None})

@app.route("/comparison/<cid>/suggestions", methods=["GET"])
def get_smart_suggestions(cid):
    """Generates context-aware queries based on the uploaded resumes."""
    session = SESSIONS.get(cid)
    if not session or not session["resumes"]:
        return jsonify({"suggestions": []})

    summary = "\n".join([f"Candidate: {r['filename']}, Content: {r['text'][:800]}" for r in session["resumes"]])
    prompt = [
        {"role": "system", "content": "Suggest 4 short, expert comparison questions based on these resumes. Return ONLY JSON: {\"suggestions\": [\"q1\", \"q2\", \"q3\", \"q4\"]}"},
        {"role": "user", "content": summary}
    ]
    
    try:
        raw = query_groq(prompt, json_mode=True)
        return jsonify(json.loads(raw))
    except:
        return jsonify({"suggestions": ["Compare technical skills", "Experience summary", "Education match", "Role fit"]})

@app.route("/ats-score", methods=["POST"])
def ats_score():
    data = request.json
    job_description = data.get("job_description")

    if not CURRENT_RESUME["text"]:
        return jsonify({"error": "Please upload a resume first."}), 400

    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    result = hybrid_ats_score(
        CURRENT_RESUME["text"],
        job_description
    )
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)