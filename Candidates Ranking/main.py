# main.py

import os
import pandas as pd
from embedding import get_embeddings
from matcher import compute_similarity, rank_jobs_for_resume
from preprocessing import get_clean_resume_text, clean_text
import openai

openai.api_key = "Add-your-key"  
# Ideally, use environment variable instead

#  1. Load Job Descriptions 
job_df = pd.read_csv("D:\protfilo\Candidate Ranking System\data\job_scrapped.csv")
job_df["clean_description"] = job_df["Description"].apply(clean_text)

#  2. Load and Process Resumes 
resumes_dir = "D:/downloads/cv/"
resume_texts = {}

for filename in os.listdir(resumes_dir):
    if filename.endswith((".pdf", ".docx")):
        path = os.path.join(resumes_dir, filename)
        resume_texts[filename] = get_clean_resume_text(path)



#  3. Generate Embeddings 
all_texts = list(job_df["clean_description"]) + list(resume_texts.values())
embeddings = get_embeddings(all_texts)

job_embeddings = embeddings[:len(job_df)]
resume_embeddings = embeddings[len(job_df):]


#  4. Match and Score 
for idx, (resume_name, resume_text) in enumerate(resume_texts.items()):
    print(f"\n🔍 Matching for: {resume_name}")
    
    scores = compute_similarity(resume_embeddings[idx], job_embeddings)
    ranked = rank_jobs_for_resume(scores, job_df)

    print(ranked[['Job Title', 'score']].head(5))