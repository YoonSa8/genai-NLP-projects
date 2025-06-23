from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaLLM
from pymongo import MongoClient
import json 

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMMA_MODEL_NAME = "gemma:3b-instruct-q4"
VECTOR_DIM = 384

with open("config.json") as f:
    config=json.load(f)

MONGODB_URI = config["MONGODB_URI"]
DB_NAME = config["DB_NAME"]
COLLECTION_NAME = config["COLLECTION_NAME"]
INDEX_NAME = config["INDEX_NAME"]

client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]


embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding= embedding_model,
    index_name=INDEX_NAME,
    text_key="text"
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs= {
        "k":3
    }
)

llm = OllamaLLM(model="gemma3:1b-it-qat")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
qa_chain.return_source_documents = True



query = "what is Gaussian mixture? and what has to do with bayssian inferance"
response = qa_chain.invoke({"query": query})


print("Answer: ")
print(response["result"])

print("\n source chunks: ")
for doc in response["source_documents"]:
    filename = doc.metadata.get("filename", "unknown_file")
    chunk_index = doc.metadata.get("chunk_index", "N/A")
    print(f"{filename} (chunk {chunk_index}):")
    print(doc.page_content[:200], "\n---")
