import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Resume Chatbot", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
.chat-container {
    max-width: 720px;
    margin: auto;
}
.user-bubble {
    background-color: #DCF8C6;
    padding: 10px 14px;
    border-radius: 18px;
    margin: 8px 0;
    text-align: right;
}
.bot-bubble {
    background-color: #F1F0F0;
    padding: 10px 14px;
    border-radius: 18px;
    margin: 8px 0;
    text-align: left;
}
.small {
    color: gray;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- State ----------
if "resume_id" not in st.session_state:
    st.session_state.resume_id = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------- Header ----------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("## 📄 Resume Chatbot")
st.markdown("<div class='small'>Upload your resume and chat with it 💬</div>", unsafe_allow_html=True)

# ---------- Upload Section ----------
st.markdown("### 📎 Upload Resume")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading resume..."):
        response = requests.post(
            f"{API_URL}/resume/upload",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            st.session_state.resume_id = response.json()["resume_id"]
            st.success("✅ Resume uploaded! You can start chatting.")
        else:
            st.error("❌ Upload failed.")

# ---------- Chat Section ----------
if st.session_state.resume_id:

    st.markdown("---")
    st.markdown("### 💬 Chat with your resume")

    # Quick chips
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🎓 CGPA"):
        st.session_state.chat.append(("user", "What is my CGPA?"))
    if col2.button("🧠 Skills"):
        st.session_state.chat.append(("user", "What skills do I have?"))
    if col3.button("📄 Summary"):
        st.session_state.chat.append(("user", "Summarize my resume"))
    if col4.button("🏢 Experience"):
        st.session_state.chat.append(("user", "What experience do I have?"))

    # Input box
    user_input = st.text_input("Type your question here...", placeholder="Ask anything about your resume...")

    if st.button("Send 💬") and user_input:
        st.session_state.chat.append(("user", user_input))

    # Process latest unanswered user message
    if st.session_state.chat and st.session_state.chat[-1][0] == "user":
        question = st.session_state.chat[-1][1]

        with st.spinner("Thinking... 🤔"):
            r = requests.post(
                f"{API_URL}/resume/ask",
                params={
                    "resume_id": st.session_state.resume_id,
                    "question": question
                }
            )

        if r.status_code == 200:
            answer = r.json()["answer"]
        else:
            answer = "Oops 😅 something went wrong."

        st.session_state.chat.append(("bot", answer))

    # Display chat history
    for role, text in st.session_state.chat:
        if role == "user":
            st.markdown(f"<div class='user-bubble'>🧑 {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>🤖 {text}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
