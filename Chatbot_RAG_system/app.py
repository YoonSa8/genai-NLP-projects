import streamlit as st
from pymongo import MongoClient
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaLLM
import json
import fitz

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMMA_MODEL_NAME = "gemma3:1b-it-qat"
VECTOR_DIM = 384
with open("config.json") as f:
    config = json.load(f)

MONGODB_URI = config["MONGODB_URI"]
DB_NAME = config["DB_NAME"]
COLLECTION_NAME = config["COLLECTION_NAME"]
INDEX_NAME = config["INDEX_NAME"]


@st.cache_resource
def init_rag_chain():
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name=INDEX_NAME,
        text_key="text"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    llm = OllamaLLM(model=GEMMA_MODEL_NAME, temperature=0.7,
                    top_p=0.95,
                    num_ctx=2048,
                    stop=None,
                    repeat_penalty=1.1)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    qa_chain.return_source_documents = True

    return qa_chain, vectorstore, collection, embedding_model


qa_chain, vectorstore, collection, embedding_model = init_rag_chain()

st.set_page_config(page_title="Gemma RAG Chatbot", layout="wide")
st.title("RAG Chatbot using my notes")
st.write("Ask questions based on your uploaded `.txt` notes.")

st.sidebar.header("Upload New Note")

uploaded_option = st.sidebar.radio("Choose format: ", ["TXT", "PDF"])

uploaded_file = st.sidebar.file_uploader(
    f"Upload a {uploaded_option} file",
    type=["txt"] if uploaded_option == "TXT" else ["pdf"]
)

if uploaded_file:
    note_name = uploaded_file.name
    file_bytes = uploaded_file.read()

    if uploaded_option == "TXT":
        content = file_bytes.decode("utf-8")

    else:  # PDF
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            content = "\n".join(page.get_text() for page in doc)
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            st.stop()

    # Split & embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)

    embeddings = embedding_model.embed_documents(chunks)

    for i, chunk in enumerate(chunks):
        doc = {
            "text": chunk,
            "embedding": embeddings[i],
            "source": note_name,
            "chunk_id": f"{note_name}_chunk{i}",
            "metadata": {
                "filename": note_name,
                "chunk_index": i,
                "created_time": datetime.datetime.now().isoformat(),
                "format": uploaded_option
            }
        }
        collection.insert_one(doc)

    st.sidebar.success(f"Uploaded and added '{note_name}' to RAG system")
    qa_chain.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

if "history" not in st.session_state:
    st.session_state.history = []

# === Display previous Q&A ===
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a["result"])
        with st.expander("📚 Source Chunks"):
            for doc in a["source_documents"]:
                meta = doc.metadata
                st.markdown(
                    f"**{meta.get('filename', 'unknown')}** - *chunk {meta.get('chunk_index', 'N/A')}*")
                st.code(doc.page_content[:500])

# === Input at bottom of screen ===
query = st.chat_input("Type your question here...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": query})

    with st.chat_message("assistant"):
        st.markdown(response["result"])
        with st.expander("📚 Source Chunks"):
            for doc in response["source_documents"]:
                meta = doc.metadata
                st.markdown(
                    f"**{meta.get('filename', 'unknown')}** - *chunk {meta.get('chunk_index', 'N/A')}*")
                st.write(doc.page_content[:200] + "\n---")
                st.code(doc.page_content[:500])

    # Save to history
    st.session_state.history.append((query, response))
