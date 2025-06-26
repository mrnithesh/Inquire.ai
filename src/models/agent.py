from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.tools.retrieval_tools import get_retrieval_tool, get_web_search_tool
from src.tools.analysis_tools import summarize, extract_kpis, generate_report

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
    system_template = """You are Inquire.ai, an autonomous research assistant specialized in analyzing documents and creating insights.

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