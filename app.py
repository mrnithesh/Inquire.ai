import streamlit as st
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma  
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports for newer implementation
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Add Tavily API key setup
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
# You can also add this to Streamlit secrets or UI input if preferred

CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Create a new project
def create_project(name):
    path = os.path.join(CHROMA_DIR, name)
    os.makedirs(path, exist_ok=True)

# List all projects
def list_projects():
    return [d for d in os.listdir(CHROMA_DIR) if os.path.isdir(os.path.join(CHROMA_DIR, d))]

# Process and chunk documents
def load_and_chunk(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Get vectorstore for a project
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

# --- Tool definition for LangGraph ---
def get_retrieval_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    @tool("retrieve_project_context")
    def retrieve(query: str) -> str:
        """Retrieve relevant context from the project documents based on the query."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found for this query."
        # Return a single string for the LLM to use as context
        return "\n\n".join(doc.page_content for doc in docs)
    
    return retrieve

# --- New Tavily web search tool ---
def get_web_search_tool():
    search_tool = TavilySearchResults(k=3)
    
    @tool("search_web")
    def search_web(query: str) -> str:
        """Search the web for current information not found in the project documents."""
        try:
            results = search_tool.invoke(query)
            if not results:
                return "No relevant information found on the web for this query."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                formatted_results.append(f"Result {i}:\nTitle: {title}\nContent: {content}\nSource: {url}\n")
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching the web: {str(e)}"
    
    return search_web

# --- Agent setup using LangGraph ---
def get_agent_executor(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    retrieve_tool = get_retrieval_tool(vectorstore)
    web_search_tool = get_web_search_tool()
    tools = [retrieve_tool, web_search_tool]
    
    # Define a custom system message template for the agent
    system_template = """You are a helpful assistant for answering questions about documents and general knowledge.
    
    To answer the user's questions, you have access to these tools:
    
    1. retrieve_project_context: Use this tool to search for relevant information in the project documents based on the user's question.
    2. search_web: Use this tool to search the web for information not found in the project documents.
    
    Follow these steps for EVERY user question:
    1. FIRST, use the retrieve_project_context tool with the user's question as the query
    2. If the retrieved context fully answers the question, synthesize a direct answer based on that information
    3. If the retrieved context partially answers or doesn't answer the question, use the search_web tool to find additional information
    4. Synthesize a comprehensive answer using both project documents and web search results as appropriate
    5. Clearly indicate which parts of your answer come from project documents versus web search
    
    IMPORTANT: NEVER ask the user to provide a query - use their original question as the query for the tools.
    
    User's question: {input}
    """
    
    # Create the agent with our tools
    agent = create_react_agent(llm, tools)
    
    # Function to properly invoke the agent with our custom approach
    def invoke_agent_with_query(query, chat_history=None):
        # Format the query to explicitly instruct the agent
        formatted_query = f"Search for information about: {query}"
        
        # Create messages to send to the agent
        messages = []
        
        # Add system message with clear instructions
        messages.append(SystemMessage(content=system_template.format(input=query)))
        
        # Add chat history if available
        if chat_history:
            messages.extend(chat_history)
        
        # Add the current query
        messages.append(HumanMessage(content=formatted_query))
        
        # Invoke the agent with these messages
        result = agent.invoke({"messages": messages})
        return result
    
    # Return our custom invoker function
    return invoke_agent_with_query

# --- UI ---
st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("RAG Assistant")


# Project selection/creation
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
        project_key = f"chat_history_{project}"
        if project_key in st.session_state:
            del st.session_state[project_key]



# --- Chat history state ---
project_key = f"chat_history_{st.session_state.get('project', 'default')}"
if project_key not in st.session_state:
    st.session_state[project_key] = []

st.header(f"Ask questions about your documents in '{st.session_state.get('project', 'N/A')}'")

# Display chat history
for message in st.session_state[project_key]:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)

query = st.chat_input("Enter your question")

if query:
    if "project" not in st.session_state:
        st.error("Please select or create a project first.")
    else:
            
        project = st.session_state["project"]
        vectorstore = get_vectorstore(project)
        agent_invoker = get_agent_executor(vectorstore)

        st.chat_message("user").write(query)
        
        # Add user message to chat history as proper LangChain Message object
        user_message = HumanMessage(content=query)
        st.session_state[project_key].append(user_message)
        
        with st.spinner("Thinking..."):
            try:
                # Use our custom invoker function that properly formats the query
                result = agent_invoker(query, st.session_state[project_key])
                
                # Extract the assistant's response
                if "messages" in result:
                    assistant_messages = [msg for msg in result["messages"] if msg.type == "ai"]
                    if assistant_messages:
                        answer = assistant_messages[-1].content
                    else:
                        answer = "I couldn't generate a proper response."
                else:
                    answer = "I couldn't find an answer based on the available documents."
                
                st.chat_message("assistant").write(answer)
                
                # Add assistant response to chat history
                assistant_message = AIMessage(content=answer)
                st.session_state[project_key].append(assistant_message)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state[project_key].append(AIMessage(content=f"I encountered an error: {error_msg}"))