import numpy as np
import pandas as pd
from typing import List


def compute_similarity(resume_embedding: List[float], job_embeddings: List[List[float]]) -> List[float]:
    #Compute cosine similarity between a single resume embedding and a list of job embeddings.
    resume_vec = np.array(resume_embedding)
    job_matrix = np.array(job_embeddings)

    # Normalize vectors
    resume_norm = resume_vec / np.linalg.norm(resume_vec)
    job_norms = job_matrix / np.linalg.norm(job_matrix, axis=1, keepdims=True)

    # Compute cosine similarities
    similarities = np.dot(job_norms, resume_norm)

    return similarities.tolist()


def rank_jobs_for_resume(similarity_scores: List[float], job_df: pd.DataFrame) -> pd.DataFrame:
    #Rank jobs based on similarity scores and return a DataFrame with scores.
    ranked_jobs = job_df.copy()
    ranked_jobs['score'] = similarity_scores
    ranked_jobs.sort_values(by='score', ascending=False, inplace=True)
    return ranked_jobs.reset_index(drop=True)
