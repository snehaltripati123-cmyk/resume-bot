📄 AI-Powered Resume Chatbot

A RAG (Retrieval-Augmented Generation) application that allows users to upload resumes (PDF, DOCX, Images) and chat with them instantly to extract skills, experience, and candidate details.

![Project Screenshot](https://via.placeholder.com/800x400?text=App+Screenshot+Here) 
*(Replace this link with your actual screenshot, e.g., image_69d662.png)*

🚀 Features

* **Multi-Format Support:** Upload PDFs, Word documents (.docx), and Images (JPG/PNG).
* **Instant Analysis:** Extracts text using OCR (Tesseract) and document parsers.
* **AI Chat Interface:** Chat with any resume using **Llama-3.3-70b** (via Groq API).
* **Context Aware:** The AI remembers the context of the specific resume you are chatting with.
* **Chat History:** Sidebar navigation to switch between different candidates/resumes.
* **Responsive UI:** Clean, modern interface built with Vanilla JS and CSS.

🛠️ Tech Stack

* **Backend:** Python, Flask
* **Database:** PostgreSQL (with SQLAlchemy ORM)
* **AI Model:** Llama-3.3-70b (via Groq Cloud)
* **Text Extraction:** `pypdf`, `python-docx`, `pytesseract` (OCR)
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

⚙️ Installation & Setup

1. Clone the Repository
```bash
git clone [https://github.com/snehaltripati123-cmyk/resume-bot.git](https://github.com/snehaltripati123-cmyk/resume-bot.git)
cd resume-bot
2. Create a Virtual Environment
Bash

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory and add your API keys:

Code snippet

GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost/resume_db
5. Setup Database
Ensure PostgreSQL is running and you have created a database named resume_db. The application will automatically create the required tables on the first run.

6. Run the Application
Bash

python app.py
Visit http://127.0.0.1:5000 in your browser.

📂 Project Structure
resume-bot/
├── app.py              # Main Flask application
├── models.py           # Database Schema (SQLAlchemy)
├── groq_service.py     # AI Integration Logic
├── resume_parser.py    # Text extraction logic (PDF/OCR)
├── database.py         # DB Connection setup
├── templates/
│   └── index.html      # Frontend UI
├── static/             # CSS/JS assets (if any)
└── uploads/            # Temporary storage for uploaded files
🔮 Future Improvements
Vector Embeddings: Implement pgvector for semantic search across all resumes.

Analytics Dashboard: Visual charts for comparing candidate experience and skills.

Export: Ability to export chat summaries as PDF.

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request.
