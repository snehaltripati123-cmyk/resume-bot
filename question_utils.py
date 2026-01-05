import re

def normalize_question(q: str) -> str:
    q = q.lower()

    replacements = {
        "exp": "experience",
        "yrs": "years",
        "edu": "education",
        "cgpa": "gpa",
        "grad": "graduation",
        "proj": "projects"
    }

    for k, v in replacements.items():
        q = re.sub(rf"\b{k}\b", v, q)

    return q

def detect_intent(q: str) -> str:
    q = q.lower()

    if "summary" in q or "overview" in q:
        return "summary"
    if "skill" in q:
        return "skills"
    if "experience" in q:
        return "experience"
    if "education" in q or "study" in q:
        return "education"
    return "qa"
