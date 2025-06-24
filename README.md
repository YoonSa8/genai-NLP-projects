# genai-NLP-projects
# 🧠 Candidate Ranking System using NLP Embeddings

This project ranks resumes against job descriptions using **local language models** and **vector similarity**. It is part of the broader [`genai-NLP-projects`](https://github.com/your-username/genai-NLP-projects) repository demonstrating applied NLP in real-world scenarios.

---

## 🚀 Features

- 📄 Parse and clean resume documents (`.pdf` and `.docx`)
- 📃 Load and clean job descriptions (scraped separately)
- 🔍 Embed both resumes and jobs using local Sentence Transformers
- 📊 Rank top job matches for each candidate (or vice versa)
- 🧪 Easily extensible to other ranking or retrieval use cases

---

## 🛠️ Tech Stack

- `sentence-transformers` – Local semantic embedding
- `pandas` – Data handling
- `scikit-learn` – Cosine similarity
- `PyMuPDF` + `python-docx` – Resume text extraction

---

## 📂 Folder Structure

candidate-ranking/
│
├── main.py # Orchestrates the workflow
├── preprocessing.py # Resume & job text extraction and cleaning
├── embedding_utils.py # Embedding generation with sentence-transformers
├── similarity_utils.py # Similarity scoring and ranking
├── data/
│ ├── jobs.csv
│ └── resumes/ # Resume files (.pdf, .docx)
└── output/
└── rankings.csv # (Optional) Output of ranked results

Each functionality is modular and reusable:

extract_resume_text() → from preprocessing.py

get_embeddings() → from embedding_utils.py

compute_similarity() & rank_jobs_for_resume() → from similarity_utils.py







# 🧠 Chatbot RAG using Gemma, HuggingFace, and MongoDB

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload their own `.txt` or `.pdf` notes and ask questions based on the uploaded content. It uses **local embedding models**, **Gemma via Ollama**, and **MongoDB Atlas** for vector search.

---

## 🚀 Features
💬 Ask questions about your documents and get AI-powered answers

🔎 Uses MongoDB Atlas vector search for fast similarity-based retrieval

🧠 Answers powered by Gemma 3B QAT4 (via Ollama) — fully local

📄 Supports uploading .txt and .pdf files to expand your knowledge base

📚 Shows source document chunks used for each answer

⚡ Embeds documents using Hugging Face’s lightweight MiniLM model

🖥️ Streamlit UI with chat-style input at the bottom of the screen



---

🧰 Tech Stack

Component	Tech

Vector Store	MongoDB Atlas (Vector Search)
Embedding Model	HuggingFace all-MiniLM-L6-v2
LLM	gemma:3b-instruct-q4 via Ollama
Frontend	Streamlit
File Support	.txt, .pdf
RAG Framework	LangChain
---

## 🛠️ Stack

| Component            | Technology                          |
|----------------------|--------------------------------------|
| Embeddings           | `HuggingFaceEmbeddings` (BGE models) |
| LLM                  | `Gemma` via `Ollama`                 |
| Vector Database      | `MongoDB Atlas`                      |
| UI                   | `Streamlit`                          |
| File Parsing         | `PyMuPDF` (for PDFs)                 |

---

## 📂 Project Structure

chatbot_rag/
│
├── app.py # Streamlit UI
├── rag.py # RAG logic with MongoDB and Ollama
├── embedding_insertDB.py # Embeds & stores uploaded notes
├── config.json # Project configuration
├── requirements.txt
└── README.md


📦 Installation
Clone the repo:
git clone https://github.com/yourusername/genai-NLP-projects.git
cd genai-NLP-projects/chatbot_rag

Install dependencies:
pip install -r requirements.txt

Run Ollama and pull model:
ollama pull gemma:2b

Launch the app:
streamlit run app.py

📝 Notes
Ollama must be installed and running: https://ollama.com
Ensure MongoDB Atlas vector search is enabled.
HuggingFace models can be changed in config.json.

📚 Example Query
Upload a chapter1.txt file with RNN content and ask:
"What is a recurrent neural network?"
The app will search the relevant chunks and generate an answer using Gemma.


📜 License
MIT License. Free for personal and educational use.

🙋‍♀️ Author
Safia Hamdy
Data Scientist & ML Developer
🔗 LinkedIn https://www.linkedin.com/in/safia-hamdy-1273a5249

---

Let me know if you'd like to:
- Split README into sections for different tools (Mongo, HuggingFace, Ollama)
- Localize it in Arabic
- Add screenshots or a demo GIF
- Publish it on Hugging Face Spaces or Render

I'm happy to help!









