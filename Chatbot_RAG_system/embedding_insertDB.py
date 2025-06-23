from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import json

with open("config.json") as f:
    config=json.load(f)

MONGODB_URI = config["MONGODB_URI"]
DB_NAME = config["DB_NAME"]
COLLECTION_NAME = config["COLLECTION_NAME"]
INDEX_NAME = config["INDEX_NAME"]

MODEL_NAME = "all-MiniLM-L6-v2"

with open("processed_notes.json", "r", encoding="utf-8") as f:
    processed_docs = json.load(f)

client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]


embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def embed_and_store(processed_docs):
    texts = [doc["text"] for doc in processed_docs]
    print(f'generating embedding for {len(texts)} chunk')

    embeddings = embedding_model.embed_documents(texts)
    print("embedding created")

    for i, doc in enumerate(processed_docs):
        mongo_doc = {
            "text": doc["text"],
            "embedding": embeddings[i],
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "metadata": doc["meta_data"]

        }
        collection.insert_one(mongo_doc)
    print("all documents are inserted")


embed_and_store(processed_docs)
