from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(context, question):
    """
    Standard fast chat.
    """
    # Truncate context to safe limit (approx 20k chars)
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
    Generates a short title for the sidebar.
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