import os
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json 

NOTE_DIR = "my_notes"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE , chunk_overlap = CHUNK_OVERLAP)


def process_notes(notes_dir):
    documents = []
    for filen in os.listdir(notes_dir): 
        if filen.endswith(".txt"):
            filepath= os.path.join(notes_dir, filen)
            with open(filepath, "r", encoding= "utf-8") as file:
                raw_txt = file.read()
            chunks = text_splitter.split_text(raw_txt)
            source_name = os.path.splitext(filen)[0]
            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "embedding": None,
                    "source": filen,
                    "chunk_id": f"{source_name}_chunk{i}",
                    "meta_data": {
                        "filename": filen,
                        "chunk_index": i,
                        "created_time": datetime.datetime.now().isoformat()
                    }
                }
                documents.append(doc)
    return documents


if __name__ == "__main__":
    all_docs = process_notes(NOTE_DIR)
    print(f"processed {len(all_docs)} chunks from .txt notes.")
    print("exapmle:\n", all_docs[0])
    with open("processed_notes.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii= False, indent=2)