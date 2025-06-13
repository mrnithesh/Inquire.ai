import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.constants import CHROMA_DIR

def create_project(name):
    path = os.path.join(CHROMA_DIR, name)
    os.makedirs(path, exist_ok=True)

def list_projects():
    return [d for d in os.listdir(CHROMA_DIR) if os.path.isdir(os.path.join(CHROMA_DIR, d))]

def load_and_chunk(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def get_vectorstore(project):
    project_dir = os.path.join(CHROMA_DIR, project)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(
        persist_directory=project_dir,
        embedding_function=embeddings,
        collection_name=project
    )

def add_docs_to_vectorstore(project, docs):
    project_dir = os.path.join(CHROMA_DIR, project)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(
        persist_directory=project_dir,
        embedding_function=embeddings,
        collection_name=project
    )
    vectordb.add_documents(docs) 