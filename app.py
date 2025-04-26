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

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

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
# --- Document summarization tool ---
@tool("summarize")
def summarize(content: str) -> str:
    """Summarize a section or full document.
    The content parameter should be the text you want to summarize."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    
    prompt = f"""
    Please provide a comprehensive yet concise summary of the following content:
    
    {content}
    
    Focus on the key points, main arguments, and important findings. 
    Organize the summary in a coherent structure with clear sections if appropriate.
    """
    
    result = llm.invoke(prompt)
    return result.content

# --- KPI extraction tool ---
@tool("extract_kpis")
def extract_kpis(content: str) -> str:
    """Extract Key Performance Indicators (KPIs) or numeric metrics from the content.
    The content parameter should be the text from which you want to extract metrics."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    prompt = f"""
    Please extract all key performance indicators (KPIs) and numeric metrics from the following content:
    
    {content}
    
    For each KPI or metric found, provide:
    1. The metric name/type
    2. The numeric value
    3. The time period or date it refers to (if available)
    4. The context around this metric
    
    Format the results as a structured list with clear categories. If you identify trends or year-over-year
    comparisons, highlight those specifically.
    """
    
    result = llm.invoke(prompt)
    return result.content

# --- Report generation tool ---
@tool("generate_report")
def generate_report(topic: str, context: str) -> str:
    """Create a brief report based on retrieved information.
    The topic parameter should be the main subject of the report.
    The context parameter should contain the relevant information to include in the report."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    prompt = f"""
    Generate a professional, concise report on the topic: "{topic}"
    
    Base your report on the following information:
    
    {context}
    
    Your report should include:
    1. An executive summary
    2. Key findings organized by relevant categories
    3. Conclusions and implications if applicable
    
    Format the report professionally with clear section headings and a logical structure.
    Maintain an objective, analytical tone throughout.
    """
    
    result = llm.invoke(prompt)
    return result.content

# --- Agent setup using LangGraph ---
def get_agent_executor(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    
    # Gather all tools
    retrieve_tool = get_retrieval_tool(vectorstore)
    web_search_tool = get_web_search_tool()
    
    # Create a list of all tools
    tools = [
        retrieve_tool, 
        web_search_tool,
        summarize,
        extract_kpis,
        generate_report
    ]
    
    # Define a custom system message template for the agent
    system_template = """You are an autonomous research assistant specialized in analyzing documents and creating insights.

You have access to these tools:

1. retrieve_project_context: Use this to search for information in the project documents. ALWAYS use this first to get relevant context.
2. search_web: Use this to search the web for information not found in the documents, especially for current events and external data.
3. summarize: Use this to create a concise summary of a document or section. Feed the retrieved content to be summarized.
4. extract_kpis: Use this to identify and extract key metrics, numbers, and KPIs from documents. Feed the retrieved content to extract from.
5. generate_report: Use this to create a professional report on a specific topic using the context you've gathered.

DECISION MAKING PROCESS:
1. FIRST, analyze the user's request carefully to determine their intent and required information.
2. Use retrieve_project_context FIRST to gather relevant information from uploaded documents.
3. If project documents don't contain sufficient information, use search_web to find complementary data.
4. Apply the appropriate analytical tools based on the user's needs:
   - For summarization requests → Use summarize()
   - For metric/KPI analysis → Use extract_kpis()
   - For  reports → Use generate_report()
5. Chain multiple tools together when needed. For example:
   - retrieve_project_context → summarize → generate_report
   - retrieve_project_context → extract_kpis + search_web → generate_report

Examples:
- If asked to "Summarize the ESG risks in the uploaded report" → Use retrieve_project_context to find ESG content, then summarize the results
- If asked to "Compare carbon emissions between 2022 and 2023" → Use retrieve_project_context with specific year queries, then extract_kpis on both sets of results
- If asked for a comprehensive analysis → Chain multiple tools and synthesize the findings

IMPORTANT: Always show your reasoning about which tools to use and why. Make autonomous decisions about the best approach to fulfill the user's request.

User's question: {input}
"""
    
    # Create the agent with our tools
    agent = create_react_agent(llm, tools)
    
    # Function to properly invoke the agent with our custom approach
    def invoke_agent_with_query(query, chat_history=None):
        # Create messages to send to the agent
        messages = []
        
        # Add system message with clear instructions
        messages.append(SystemMessage(content=system_template.format(input=query)))
        
        # Add chat history if available
        if chat_history:
            messages.extend(chat_history)
        
        # Add the current query
        messages.append(HumanMessage(content=query))
        
        # Invoke the agent with these messages
        result = agent.invoke({"messages": messages})
        return result
    
    # Return our custom invoker function
    return invoke_agent_with_query

# --- UI ---
st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)
st.header("🤖 Research Assistant")
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

st.write(f"Ask questions about your documents in '{st.session_state.get('project', 'N/A')}'")

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
                
                st.chat_message("assistant").write(answer)
                
                # Add assistant response to chat history
                assistant_message = AIMessage(content=answer)
                st.session_state[project_key].append(assistant_message)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state[project_key].append(AIMessage(content=f"I encountered an error: {error_msg}"))