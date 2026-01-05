import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_with_groq(question: str, context: str, intent: str) -> str:
    if intent == "summary":
        task = "Provide a structured summary of the resume including education, experience, skills, and projects."
    elif intent == "skills":
        task = "List the skills mentioned in the resume."
    elif intent == "experience":
        task = "Summarize the work experience from the resume."
    elif intent == "education":
        task = "Summarize the education background from the resume."
    else:
        task = "Answer the question using only the resume."

    prompt = f"""
You are a resume assistant.

Rules:
- Only use information from the resume context.
- Do not hallucinate or add information.
- If the information is not present, say: "This information is not present in the resume."

Task:
{task}

Context:
{context}

Question:
{question}

Answer:
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return completion.choices[0].message.content.strip()
