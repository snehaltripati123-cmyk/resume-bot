"""
Dynamic ATS Engine — Groq does skill extraction AND match judgment.
No hardcoded skill lists. Works for any domain.

The key insight: let the LLM decide what "matches" — it understands that
Git = version control, NLP = natural language processing, production ML = MLOps, etc.
Sentence transformers handle the overall semantic similarity score only.

Install:
    pip install groq sentence-transformers nltk rapidfuzz
"""

import re
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading models...")
_sentence_model = SentenceTransformer('all-mpnet-base-v2')
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# BOILERPLATE STRIPPING
# --------------------------------------------------

_BOILERPLATE_PATTERNS = [
    r'equal opportunity employer[^.]*\.',
    r'we (do not|don\'t) discriminate[^.]*\.',
    r'without regards? to (race|color|creed|religion|sex|national origin|age|marital status|veteran|disability|genetic|sexual orientation)[^.]*\.',
    r'diversity and inclusion are[^.]*\.',
    r'we (seek|strive|are committed) to (recruit|develop|retain)[^.]*\.',
    r'employment (opportunities|decisions)[^.]*\.',
    r'protected by (law|federal|state)[^.]*\.',
    r'salary range[^.]*\.',
    r'competitive (salary|compensation|benefits)[^.]*\.',
    r'401k[^.]*\.',
    r'apply (now|today|online)[^.]*\.',
    r'we are an? (equal|proud)[^.]*\.',
]

_BOILERPLATE_HEADERS = [
    "equal opportunity", "diversity & inclusion", "diversity and inclusion",
    "about our benefits", "compensation", "what we offer", "eeo statement",
    "non-discrimination", "reasonable accommodation", "background check"
]

_JD_SIGNAL_WORDS = [
    "experience", "required", "preferred", "qualifications", "responsibilities",
    "skills", "proficient", "familiar", "bachelor", "master", "degree", "years",
    "ability", "must", "develop", "design", "build", "manage", "lead"
]

def strip_boilerplate(jd_text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        jd_text = re.sub(pattern, ' ', jd_text, flags=re.IGNORECASE)

    lines = jd_text.split('\n')
    cleaned_lines = []
    skip_section = False

    for line in lines:
        line_lower = line.lower().strip()
        if any(h in line_lower for h in _BOILERPLATE_HEADERS):
            skip_section = True
            continue
        if skip_section and len(line.strip()) < 60 and line.strip().endswith(':'):
            skip_section = False
        if not skip_section:
            cleaned_lines.append(line)

    return re.sub(r'\s+', ' ', '\n'.join(cleaned_lines)).strip()


def validate_jd(jd_text: str) -> dict:
    if len(jd_text.strip()) < 50:
        return {
            "valid": False,
            "warning": "Job description is too short. Please paste the full JD.",
            "cleaned_jd": jd_text
        }

    cleaned = strip_boilerplate(jd_text)
    cleaned_lower = cleaned.lower()
    signal_count = sum(1 for w in _JD_SIGNAL_WORDS if w in cleaned_lower)
    meaningful = [w for w in re.findall(r'\b[a-z]{3,}\b', cleaned_lower) if w not in _stop_words]

    if len(meaningful) < 30:
        return {
            "valid": False,
            "warning": (
                "This doesn't appear to contain job requirements. "
                "Please paste the full JD including responsibilities and required skills."
            ),
            "cleaned_jd": cleaned
        }

    warning = None
    if signal_count < 3:
        warning = (
            "⚠️ This JD may only contain boilerplate. "
            "For an accurate score, include the requirements and qualifications section."
        )

    return {"valid": True, "warning": warning, "cleaned_jd": cleaned}


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()

def preprocess(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    tokens = [_lemmatizer.lemmatize(w) for w in text.split()
              if w not in _stop_words and len(w) > 1]
    return " ".join(tokens)


# --------------------------------------------------
# CORE: GROQ DOES EXTRACTION + MATCHING IN ONE CALL
# --------------------------------------------------

def groq_analyze(resume_text: str, jd_text: str) -> dict:
    """
    Single Groq call that:
    1. Extracts all requirements from the JD
    2. For each requirement, reasons whether the resume meets it —
       including synonyms, implied skills, equivalent experience

    This is the key fix: the LLM understands that:
    - "Git" in resume → satisfies "version control" in JD
    - "NLP" → satisfies "natural language processing"
    - "model drift detection + production workflows" → satisfies "MLOps"
    - "collaborated with stakeholders" → satisfies "communication"
    - "PyTorch pipelines in production" → implies "cloud environment" partially
    """

    prompt = f"""You are a senior technical recruiter and ATS specialist with deep domain expertise.

Analyze this resume against the job description and determine which requirements are met.

--- JOB DESCRIPTION ---
{jd_text[:4000]}

--- RESUME ---
{resume_text[:4000]}

Your task:
1. Extract every distinct skill, tool, qualification, and requirement from the JD (be thorough — 15-35 items)
2. For each requirement, determine if the resume satisfies it — using intelligent matching:
   - Synonyms count: "Git" satisfies "version control", "NLP" satisfies "natural language processing"
   - Implied skills count: production ML deployment implies some MLOps knowledge
   - Related tools count: PyTorch/TensorFlow experience implies ML framework familiarity
   - Broader category matches: if JD says "cloud platforms (AWS, Azure)" and resume shows cloud work, partial match counts
   - Academic/project experience counts for intern/junior roles
   - Ecosystem implications count: pandas users almost always use numpy (they're inseparable); scikit-learn implies numpy; any data science work implies basic numpy/pandas familiarity
   - Visualization ecosystem: if resume mentions "visualization tools", plotting, dashboards, EDA, or libraries like plotly/seaborn/matplotlib — treat ALL common viz libraries (matplotlib, seaborn, plotly) as implied matched
   - If someone lists a higher-level tool, the foundational dependency is implied (e.g. PyTorch → numpy, Hugging Face → PyTorch, LangChain → Python)
   - "exploratory data analysis" or "EDA" directly implies matplotlib and seaborn usage
3. Be FAIR and GENEROUS for intern/entry-level roles — if they show relevant exposure, count it as matched

Respond ONLY with valid JSON, nothing else:

{{
  "jd_requirements": [
    {{
      "skill": "exact requirement from JD",
      "matched": true or false,
      "reason": "one sentence explaining why matched or not"
    }}
  ],
  "summary": "2-3 sentence overall assessment of fit"
}}

Be thorough in extraction but fair in matching. Do not penalize for exact wording differences.
"""

    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return data

    except Exception as e:
        print(f"Groq error: {e}")
        return {"jd_requirements": [], "summary": "Analysis failed."}


def parse_groq_results(groq_data: dict) -> tuple:
    """Parse Groq output into matched/missing lists."""
    requirements = groq_data.get("jd_requirements", [])

    matched = []
    missing = []

    for req in requirements:
        skill = req.get("skill", "").lower().strip()
        if not skill:
            continue
        if req.get("matched", False):
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


# --------------------------------------------------
# SUPPORTING SCORES (semantic + keyword density)
# --------------------------------------------------

def embedding_similarity(resume_text: str, jd_text: str) -> float:
    """Overall semantic similarity — chunked to avoid dilution."""
    def prep(t):
        t = clean_text(t).lower()
        t = re.sub(r'[^a-z0-9\s\+\#]', ' ', t)
        tokens = [_lemmatizer.lemmatize(w) for w in t.split()
                  if w not in _stop_words and len(w) > 1]
        return " ".join(tokens)

    resume_clean = prep(resume_text)
    jd_clean = prep(jd_text)

    chunks = [resume_clean[i:i+512] for i in range(0, len(resume_clean), 512)]
    if not chunks:
        return 0.0

    resume_embs = _sentence_model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    jd_emb = _sentence_model.encode(jd_clean, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(jd_emb, resume_embs)[0]
    return round(float(sims.max()) * 100, 2)


def keyword_density_score(resume_text: str, jd_text: str) -> float:
    """Fraction of meaningful JD keywords present in resume."""
    generic = {
        "the","and","for","with","this","that","have","will","are","you",
        "our","your","they","from","must","able","work","team","role",
        "join","good","strong","looking","candidate","position","company",
        "opportunity","experience","required","preferred","responsibilities",
        "qualifications","who","what","you'll","we're","they'll"
    }

    def words(text):
        return set(re.findall(r'\b[a-z][a-z0-9\+\#]{2,}\b', text.lower())) - generic

    jd_words = words(jd_text)
    resume_words = words(resume_text)

    if not jd_words:
        return 50.0

    overlap = jd_words & resume_words
    return round(min((len(overlap) / len(jd_words)) * 100, 100), 2)


def extract_years(text: str) -> int:
    matches = re.findall(r'(\d+)\+?\s*(?:to\s*\d+\s*)?year', text.lower())
    return max((int(m) for m in matches), default=0)

def experience_score(resume_text: str, jd_text: str) -> float:
    jd_years = extract_years(jd_text)
    resume_years = extract_years(resume_text)

    if jd_years == 0:
        return 75.0
    if resume_years >= jd_years:
        return 100.0
    elif resume_years == 0:
        return 60.0  # no explicit years but may still be qualified
    return round((resume_years / jd_years) * 100, 2)


# --------------------------------------------------
# FINAL HYBRID ATS SCORE
# --------------------------------------------------

def hybrid_ats_score(resume_text: str, jd_text: str) -> dict:
    """
    Compute ATS score. Groq LLM handles intelligent skill matching.
    Sentence transformers add semantic similarity signal.

    Weights:
        55% — LLM skill match (intelligent, context-aware)
        20% — Keyword density (raw token overlap)
        15% — Semantic embedding similarity
        10% — Experience match
    """
    resume_text = clean_text(resume_text)

    # Validate JD first
    jd_check = validate_jd(jd_text)
    cleaned_jd = jd_check["cleaned_jd"]

    if not jd_check["valid"]:
        return {
            "ATS Score": 0,
            "error": jd_check["warning"],
            "Matched Skills": [],
            "Missing Skills": [],
            "Skill Match Score": 0,
            "Keyword Density Score": 0,
            "Semantic Score": 0,
            "Experience Score": 0,
            "Summary": ""
        }

    # Core LLM analysis
    groq_data = groq_analyze(resume_text, cleaned_jd)
    matched, missing = parse_groq_results(groq_data)

    total = len(matched) + len(missing)
    skill_score = round((len(matched) / total) * 100, 2) if total > 0 else 50.0

    # Supporting scores
    exp_score = experience_score(resume_text, jd_text)
    sem_score = embedding_similarity(resume_text, cleaned_jd)
    kw_score = keyword_density_score(resume_text, cleaned_jd)

    final_score = (
        skill_score * 0.55 +
        kw_score    * 0.20 +
        sem_score   * 0.15 +
        exp_score   * 0.10
    )

    result = {
        "ATS Score":             round(float(final_score), 2),
        "Skill Match Score":     skill_score,
        "Matched Skills":        matched,
        "Missing Skills":        missing,
        "Keyword Density Score": kw_score,
        "Semantic Score":        sem_score,
        "Experience Score":      exp_score,
        "Summary":               groq_data.get("summary", "")
    }

    # Surface any JD quality warning
    if jd_check.get("warning"):
        result["warning"] = jd_check["warning"]

    return result


# --------------------------------------------------
# TEST BLOCK
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    resume_sample = """
    Snehal Tripathi — AI/ML Engineering Intern
    Experience: AI/ML Intern at Lighthouse Info Systems — PyTorch models in production,
    preprocessing pipelines, model drift detection. AI/ML Engineering Intern at GHRCE —
    end-to-end ML solutions, data preprocessing, feature engineering, pandas, model evaluation.
    Projects: ISRO Hackathon — DEM preprocessing pipelines, drainage detection models.
    GPT Builder — secure multi-tenant GPT platform, RBAC, audit logging, vector ingestion,
    log analysis. Medical Safety LLM — LoRA fine-tuning, 4-bit quantization, safety evaluation.
    Skills: Python, SQL, Git, LLMs, RAG, Generative AI, Fine-Tuning, NLP, Computer Vision,
    PyTorch, TensorFlow, Scikit-Learn, Hugging Face, LangChain, FastAPI, Streamlit,
    Vector Databases, OpenCV, Oracle Cloud Generative AI Certified.
    Education: B.Tech CS (AI & ML), CGPA 8.92/10, expected 2026.
    """

    jd_sample = """
    CrashPlan — Intern AI/ML Engineer
    Day in the Life: MLOps and secure development best practices, data pipelines for security
    insights, anomaly detection, log analysis, data simulation, cloud environment setup,
    code reviews, sprint planning.
    Required: Python, ML frameworks, NLP, data processing, classification, Bachelor's CS/AI/ML.
    Preferred: data pipelines, deep learning, NLP, anomaly detection, AWS Bedrock/Sagemaker/
    Azure AI, AI Agents, RAG, fine-tuning, MCP Servers, cloud platforms (AWS, Azure),
    MLOps, CI/CD, Git, version control, SaaS, cybersecurity, data security.
    """

    print("Running ATS analysis...")
    result = hybrid_ats_score(resume_sample, jd_sample)

    print(f"\n{'='*50}")
    print(f"ATS Score: {result['ATS Score']}%")
    print(f"Skill Match: {result['Skill Match Score']}%")
    print(f"Keyword Density: {result['Keyword Density Score']}%")
    print(f"Semantic Score: {result['Semantic Score']}%")
    print(f"Experience Score: {result['Experience Score']}%")
    print(f"\nMatched ({len(result['Matched Skills'])}): {result['Matched Skills']}")
    print(f"\nMissing ({len(result['Missing Skills'])}): {result['Missing Skills']}")
    print(f"\nSummary: {result['Summary']}")

# import re
# import nltk
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer

# from skills_db import SKILLS   # Your curated skill list

# # --------------------------------------------------
# # LOAD MODELS
# # --------------------------------------------------

# model = SentenceTransformer('all-MiniLM-L6-v2')

# nltk.download('stopwords')
# nltk.download('wordnet')

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# # --------------------------------------------------
# # TEXT PREPROCESSING
# # --------------------------------------------------

# def preprocess(text):
#     text = text.replace("\n", " ")   # 🔥 fix broken line splits
#     text = re.sub(r'\s+', ' ', text) # normalize multiple spaces
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
#     tokens = text.split()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
#     return " ".join(tokens)

# # --------------------------------------------------
# # CLEAN JD (Remove Salary, Dates, Location etc.)
# # --------------------------------------------------

# def clean_job_description(jd_text):
#     jd_text = jd_text.lower()

#     # Remove salary, stipend, dates, months
#     jd_text = re.sub(r'\b\d+\b', '', jd_text)
#     jd_text = re.sub(r'₹|\$|rs\.?', '', jd_text)

#     # Remove common noise sections
#     noise_words = [
#         "stipend", "salary", "location", "deadline",
#         "equal opportunity", "apply now", "about us",
#         "certificate", "employment opportunity"
#     ]

#     for word in noise_words:
#         jd_text = jd_text.replace(word, "")

#     return jd_text

# # --------------------------------------------------
# # SKILL EXTRACTION
# # --------------------------------------------------

# SKILL_SYNONYMS = {
#     "machine learning": ["ml"],
#     "data analysis": ["data analytics", "analyzed data"],
#     "artificial intelligence": ["ai"],
#     "tensorflow": ["tf"],
#     "pytorch": ["torch"]
# }

# def skill_present(skill, text):
#     text = text.lower()

#     if skill in text:
#         return True

#     if skill in SKILL_SYNONYMS:
#         for synonym in SKILL_SYNONYMS[skill]:
#             if synonym in text:
#                 return True

#     return False


# def extract_skills_from_text(text):
#     found = []

#     for skill in SKILLS:
#         if skill_present(skill.lower(), text):
#             found.append(skill.lower())

#     return list(set(found))


# # --------------------------------------------------
# # SEMANTIC SKILL MATCHING
# # --------------------------------------------------

# def semantic_skill_score(resume_text, jd_text):

#     jd_text = clean_job_description(jd_text)

#     resume_skills = extract_skills_from_text(resume_text)
#     jd_skills = extract_skills_from_text(jd_text)

#     if not jd_skills:
#         return 0, [], []

#     matched = []
#     missing = []

#     resume_embedding = model.encode(resume_text)

#     for skill in jd_skills:
#         skill_embedding = model.encode(skill)

#         similarity = np.dot(resume_embedding, skill_embedding) / (
#             np.linalg.norm(resume_embedding) * np.linalg.norm(skill_embedding)
#         )

#         if similarity >= 0.60:
#             matched.append(skill)
#         else:
#             missing.append(skill)

#     score = (len(matched) / len(jd_skills)) * 100

#     return round(score, 2), matched, missing

# # --------------------------------------------------
# # EXPERIENCE MATCHING
# # --------------------------------------------------

# def extract_years(text):
#     matches = re.findall(r'(\d+)\+?\s*year', text.lower())
#     return max([int(m) for m in matches], default=0)

# def experience_score(resume_text, jd_text):

#     jd_years = extract_years(jd_text)
#     resume_years = extract_years(resume_text)

#     if jd_years == 0:
#         return 70  # neutral if JD doesn't specify experience

#     if resume_years >= jd_years:
#         return 100
#     else:
#         return (resume_years / jd_years) * 100

# # --------------------------------------------------
# # OVERALL SEMANTIC SIMILARITY
# # --------------------------------------------------

# def embedding_similarity(resume_text, jd_text):

#     jd_text = clean_job_description(jd_text)

#     resume_clean = preprocess(resume_text)
#     jd_clean = preprocess(jd_text)

#     embeddings = model.encode([resume_clean, jd_clean])

#     cosine_sim = np.dot(embeddings[0], embeddings[1]) / (
#         np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
#     )

#     return round(float(cosine_sim * 100), 2)

# # --------------------------------------------------
# # FINAL HYBRID ATS SCORE
# # --------------------------------------------------


# def hybrid_ats_score(resume_text, jd_text):
#     resume_text = resume_text.replace("\n", " ")
#     resume_text = re.sub(r'\s+', ' ', resume_text)

#     # 1️⃣ Skill match (main score)
#     skill_score, matched, missing = semantic_skill_score(resume_text, jd_text)

#     # 2️⃣ Experience score
#     exp_score = experience_score(resume_text, jd_text)

#     # 3️⃣ Overall relevance
#     sem_score = embedding_similarity(resume_text, jd_text)

#     # Final weighted score
#     final_score = (
#     skill_score * 0.7 +
#     exp_score * 0.15 +
#     sem_score * 0.15
#   )


#     return {
#         "ATS Score": round(float(final_score), 2),
#         "Skill Match Score": skill_score,
#         "Matched Skills": matched,
#         "Missing Skills": missing,
#         "Experience Score": round(float(exp_score), 2),
#         "Semantic Score": round(float(sem_score), 2)
#     }

# # --------------------------------------------------
# # TEST BLOCK
# # --------------------------------------------------

# if __name__ == "__main__":

#     resume = """
#     I have 2 years of experience in Python, Machine Learning,
#     TensorFlow and AWS.
#     """

#     jd = """
#     Must have Python and Machine Learning.
#     Required: 2+ years experience.
#     Nice to have Docker.
#     """

#     result = hybrid_ats_score(resume, jd)

#     for key, value in result.items():
#         print(f"{key}: {value}")
