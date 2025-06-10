import openai
from typing import List
import time
# You must set your OpenAI API key before using
# def get_embeddings(texts: List[str], model="text-embedding-3-small", batch_size=100) -> List[List[float]]:
#     """
#     Generate embeddings for a list of texts using OpenAI's embedding model.
#     Args:
#         texts: List of preprocessed input strings.
#         model: OpenAI embedding model (default is 'text-embedding-3-small').
#         batch_size: Number of texts to send per API call.
#     Returns:
#         List of embedding vectors (one per input).
#     """
#     embeddings = []

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         try:
#             response = openai.embeddings.create(input=batch, model=model)
#             batch_embeddings = [record.embedding for record in response.data]
#             embeddings.extend(batch_embeddings)
#         except Exception as e:
#             print(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
#             # Optional: retry after a pause
#             time.sleep(5)

#     return embeddings


#to avoid API limits, I can help you switch to using Hugging Face models
from sentence_transformers import SentenceTransformer
from typing import List

# Load a pre-trained model (this downloads it the first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate sentence embeddings using a local model.
    """
    return model.encode(texts, show_progress_bar=True).tolist()
