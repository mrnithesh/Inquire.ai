from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

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