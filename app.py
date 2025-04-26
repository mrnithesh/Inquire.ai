import streamlit as st
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma  
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    # collection_name for project isolation
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

# --- Tool definition ---
def get_retrieval_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Updated tool definition to be compatible with older LangChain versions
    from langchain.tools import Tool
    
    def retrieve(query: str) -> str:
        """Retrieve relevant context from the project documents based on the query."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found for this query."
        # Return a single string for the LLM to use as context
        return "\n\n".join(doc.page_content for doc in docs)
    
    return Tool(
        name="retrieve_project_context",
        description="Retrieve relevant context from the current project's documents to answer user questions.",
        func=retrieve
    )

# --- Agent setup ---
def get_agent_executor(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    retrieve_tool = get_retrieval_tool(vectorstore)
    tools = [retrieve_tool]
    
    # Create a system message that explicitly instructs the agent how to use the tool
    system_message = """You are a helpful assistant for answering questions about documents. 
    When the user asks a question, ALWAYS use the retrieve_project_context tool first to get relevant information.
    After receiving information from the tool, synthesize it to directly answer the user's question.
    If the retrieved context doesn't contain relevant information, tell the user that the information is not available in the documents.
    
    Do NOT ask the user for queries or additional information - use their original question to retrieve context.
    
    Format your responses clearly and directly address the user's question.
    """
    
    # For older LangChain versions
    from langchain.agents import initialize_agent, AgentType
    
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        early_stopping_method="force",
        handle_parsing_errors=True
    )
    

    try:
        agent_executor.agent.llm_chain.prompt.messages[0].prompt.template = system_message
    except:
        pass
        
    return agent_executor

# UI
st.set_page_config(page_title="RAG Assistant", layout="wide")
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
    if "project" not in st.session_state:
        st.sidebar.error("Please select or create a project first.")
    else:
        project = st.session_state["project"]
        for uploaded_file in uploaded_files:
            suffix = uploaded_file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                docs = load_and_chunk(tmp.name, "pdf" if suffix == "pdf" else "txt")
                add_docs_to_vectorstore(project, docs)
            os.unlink(tmp.name)
        st.sidebar.success("Documents processed and added to project.")
        # Optionally clear chat history when new docs are added
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]

# --- Chat history state ---
project_key = f"chat_history_{st.session_state.get('project', 'default')}"
if project_key not in st.session_state:
    st.session_state[project_key] = []

st.header(f"Ask questions about your documents in '{st.session_state.get('project', 'N/A')}'")

# Display chat history
for message in st.session_state[project_key]:
    if isinstance(message, dict):
        role = message.get("role", "")
        content = message.get("content", "")
    else:
        role = message.type if hasattr(message, "type") else ""
        content = message.content if hasattr(message, "content") else ""
    
    if role == "human":
        st.chat_message("user").write(content)
    elif role == "ai":
        st.chat_message("assistant").write(content)

query = st.chat_input("Enter your question")

if query:
    if "project" not in st.session_state:
        st.error("Please select or create a project first.")
    else:
        project = st.session_state["project"]
        vectorstore = get_vectorstore(project)
        agent_executor = get_agent_executor(vectorstore)

        st.chat_message("user").write(query)
        
        # Add user message to chat history
        st.session_state[project_key].append({"role": "human", "content": query})
        
        # Convert chat history to the format expected by the agent
        chat_history = []
        for msg in st.session_state[project_key]:
            if msg["role"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))
        
        with st.spinner("Thinking..."):
            try:
                # Pass the query directly, but include chat history for context
                result = agent_executor.invoke({
                    "input": query,
                    "chat_history": chat_history
                })
                
                answer = result.get("output", "I couldn't find an answer based on the available documents.")
                st.chat_message("assistant").write(answer)
                
                # Add assistant response to chat history
                st.session_state[project_key].append({"role": "ai", "content": answer})
                
                # Show the retrieved context in an expander
                if "intermediate_steps" in result:
                    contexts = []
                    for step in result["intermediate_steps"]:
                        if step[0].tool == "retrieve_project_context":
                            contexts.append(step[1])
                    
                    if contexts:
                        with st.expander("Retrieved Context"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Context Chunk {i+1}:**")
                                st.text(context[:1000] + ("..." if len(context) > 1000 else ""))
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state[project_key].append({"role": "ai", "content": f"I encountered an error: {error_msg}"})