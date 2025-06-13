from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

def get_retrieval_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    @tool("retrieve_project_context")
    def retrieve(query: str) -> str:
        """Retrieve relevant context from the project documents based on the query."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found for this query."
        return "\n\n".join(doc.page_content for doc in docs)
    
    return retrieve

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