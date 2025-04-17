from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import pandas as pd
import fitz

pdf_path = "./constitution.pdf"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_legal_db"
add_documents = not os.path.exists(db_location)


def load_pdf(file_path):
    """Load a PDF file and extract text from it."""
    pdf_document = fitz.open(file_path)
    text = ""
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return text


def split_text(raw_text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = splitter.split_text(raw_text)
    documents = [
        Document(page_content=chunk, metadata={"source": "contitution", "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    return documents


vector_store = Chroma(
    collection_name="Sri_Lanka_Constitution",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    full_text = load_pdf(pdf_path)
    documents = split_text(full_text)
    vector_store.add_documents(documents)
    vector_store.persist()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
