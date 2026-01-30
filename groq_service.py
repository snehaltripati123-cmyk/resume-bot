import os
import json
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(context, question):
    """
    Standard Chat Mode
    """
    prompt = f"Context: {context[:20000]}\n\nQuestion: {question}\nAnswer:"
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_chat_title(question):
    """
    Summarizes question into a title
    """
    prompt = f"Summarize this into a 3-5 word title. No quotes. Text: {question}"
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=15
        )
        return completion.choices[0].message.content.strip()
    except:
        return "Chat"

def extract_resume_data(text):
    """
    ANALYTICS MODE: Forces AI to return strict JSON for charts.
    """
    system_prompt = """
    You are a Data Extraction AI. 
    You must extract the following fields from the resume text and return them as a valid JSON object.
    Do not add any markdown formatting (like ```json). Just the raw JSON string.
    
    Fields to extract:
    1. "name": (String) Candidate Name
    2. "years_experience": (Number) Total years of work experience (estimate if needed, e.g., 2.5)
    3. "skills": (List of Strings) Top 5-7 technical skills.
    4. "education": (String) Highest degree (e.g., "B.Tech", "Masters").
    5. "role_fit": (String) A 2-word summary of their role (e.g., "Backend Dev", "Data Scientist").
    
    Example Output:
    {
        "name": "John Doe",
        "years_experience": 4.5,
        "skills": ["Python", "SQL", "Flask"],
        "education": "B.Tech",
        "role_fit": "Python Developer"
    }
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Resume Text: {text[:20000]}"}
            ],
            temperature=0.1, # Low temp for consistency
            response_format={"type": "json_object"} # FORCE JSON (Critical)
        )
        # Parse string to ensure it's valid dict
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Extraction Error: {e}")
        # Return fallback data so the chart doesn't crash
        return {
            "name": "Unknown", 
            "years_experience": 0, 
            "skills": [], 
            "education": "N/A",
            "role_fit": "Unknown"
        }