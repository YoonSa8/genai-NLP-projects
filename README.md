﻿# genai-NLP-projects
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
