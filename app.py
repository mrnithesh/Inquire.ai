import streamlit as st
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma  # updated import per deprecation warning
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA


CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)



#crate a new project
def create_project(name):
    path = os.path.join(CHROMA_DIR, name)
    os.makedirs(path, exist_ok=True)
#list all projects
def list_projects():
    return [d for d in os.listdir(CHROMA_DIR) if os.path.isdir(os.path.join(CHROMA_DIR, d))]

# process and chunk documents
def load_and_chunk(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# get vectorstore for a project
def get_vectorstore(project):
    project_dir = os.path.join(CHROMA_DIR, project)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(persist_directory=project_dir, embedding_function=embeddings)

def add_docs_to_vectorstore(project, docs):
    project_dir = os.path.join(CHROMA_DIR, project)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=project_dir, embedding_function=embeddings)
    vectordb.add_documents(docs)


# rag
def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ui
st.set_page_config(page_title=" RAG Assistant", layout="wide")
st.title("RAG Assistant")

# project selection/creation

st.sidebar.header("Projects")
projects = list_projects()
project = st.sidebar.selectbox("Select Project", projects + ["+ Create new project"])
if project == "+ Create new project":
    new_project = st.sidebar.text_input("New project name")
    if st.sidebar.button("Create") and new_project:
        create_project(new_project)
        st.rerun()
    st.stop()
else:
    st.session_state["project"] = project

# Upload documents
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            docs = load_and_chunk(tmp.name, "pdf" if suffix == "pdf" else "txt")
            add_docs_to_vectorstore(project, docs)
        os.unlink(tmp.name)
    st.sidebar.success("Documents processed and added to project.")

# query interface

st.header(f"Ask questions about your documents in '{project}'")
query = st.text_input("Enter your question")
if query:
    vectorstore = get_vectorstore(project)
    qa_chain = get_qa_chain(vectorstore)
    with st.spinner("Retrieving and answering..."):

        result = qa_chain.invoke({"query": query})
        st.markdown("**Answer:**")
        st.write(result["result"])
        st.markdown("---")
        st.markdown("**Relevant Context:**")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(doc.page_content[:1000])
