from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pdfplumber
import shutil
import time
from chromadb.config import Settings


embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://127.0.0.1:11434")

def extract_pdf_text(pdf_path):
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num + 1}
                    )
                    documents.append(doc)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return documents

db_location = "./chroma_db"

def delete_vector_store():
    if os.path.exists(db_location):
        shutil.rmtree(db_location)
    
def vectore_reset():
    vector_store = Chroma(
        collection_name="resume_collection",
        embedding_function=embeddings,
        persist_directory=db_location,
        client_settings=Settings(allow_reset=True)
    )
    vector_store._client.reset()
    #delete_vector_store()

def load_and_add_pdf(pdf_path):
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return 0
    
    vectore_reset()
    vector_store = Chroma(
        collection_name="resume_collection",
        embedding_function=embeddings,
        persist_directory=db_location,
        client_settings=Settings(allow_reset=True)
    )
    documents = extract_pdf_text(pdf_path)
    print(documents)
    if documents:
        vector_store.add_documents(documents)
        print(f"{len(documents)} documents added from {pdf_path}")
    return len(documents)

def get_retriever():
    vector_store = Chroma(
        collection_name="resume_collection",
        embedding_function=embeddings,
        persist_directory=db_location,
        client_settings=Settings(allow_reset=True)
    )
    return vector_store.as_retriever(search_kwargs={"k": 10})