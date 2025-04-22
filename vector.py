from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import pdfplumber
import shutil
embeddings = OllamaEmbeddings(model="mxbai-embed-large",base_url="http://127.0.0.1:11434")
# Load the PDF file
# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:  # Only add non-empty text
                    # Create a Document with text and metadata
                    doc = Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num + 1}
                    )
                    documents.append(doc)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return documents

pdf_path = "pdf/SURYA VENKATA SATISH BHIMAVARAPU.pdf"
db_location= "./chroma_db"

# Delete old vector store
if os.path.exists(db_location):
    try:
        shutil.rmtree(db_location)
        print(f"Old vector database deleted at {db_location}")
    except PermissionError as e:
        print(f"Error deleting vector store: {e}\nEnsure no other process is using {db_location}")
        raise



# Extract text from PDF
documents = []
documents.extend(extract_pdf_text(pdf_path))
print(documents)
# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="resume_collection",
    embedding_function=embeddings,
    persist_directory= db_location  # Directory to save the vector store
)

if documents:
    # Add documents to the vector store
    vector_store.add_documents(documents)
    print(len(documents),"added to vectordatabase")

retriver = vector_store.as_retriever(
    search_kwargs={"k":3}
)