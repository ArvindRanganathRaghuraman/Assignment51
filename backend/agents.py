from langgraph.graph import Graph
from langgraph.constants import END, START
from typing import Dict, Any, TypedDict, Optional, Literal
import logging
import time
from datetime import datetime

# Importing all agent functions
from websearch import fetch_nvidia_news
from hybrid_search_pinecone_assign5 import query_pinecone_with_gpt
from nvidia_snowflake import get_nvidia_financial_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchState(TypedDict):
    query: str
    use_snowflake: bool
    use_rag: bool
    use_web_search: bool
    year: Optional[int]
    quarter: Optional[int]
    snowflake_result: Optional[Dict[str, Any]]
    rag_result: Optional[str]
    web_results: Optional[str]
    final_report: Optional[str]
    error: Optional[str]

def validate_input(state: ResearchState) -> ResearchState:
    """Validate all input parameters including RAG and web search requirements"""
    if not state.get("query"):
        state["error"] = "Query cannot be empty"
        return state
    
    # Snowflake-specific validation
    if state["use_snowflake"]:
        current_year = datetime.now().year
        if state.get("year"):
            if not (2000 <= state["year"] <= current_year):
                state["error"] = f"Year must be between 2000 and {current_year}"
        if state.get("quarter"):
            if not (1 <= state["quarter"] <= 4):
                state["error"] = "Quarter must be between 1 and 4"
    
    # RAG-specific validation could be added here if needed
    
    if not (state["use_snowflake"] or state["use_rag"] or state["use_web_search"]):
        state["error"] = "At least one data source must be selected"
    
    return state

def snowflake_agent(state: ResearchState) -> ResearchState:
    """Query NVIDIA financial data from Snowflake"""
    if not state["use_snowflake"]:
        return state
    
    try:
        logger.info("Querying Snowflake...")
        query_parts = []
        if state.get("year"):
            query_parts.append(f"year={state['year']}")
        if state.get("quarter"):
            query_parts.append(f"quarter={state['quarter']}")
        
        input_str = ", ".join(query_parts) if query_parts else state["query"]
        response = get_nvidia_financial_response(input_str)
        
        if "error" in response:
            state["error"] = f"Snowflake error: {response['error']}"
        else:
            state["snowflake_result"] = {
                "text": response.get("output", ""),
                "chart_path": response.get("chart_path", "")
            }
            
    except Exception as e:
        logger.error(f"Snowflake query failed: {str(e)}", exc_info=True)
        state["error"] = f"Snowflake error: {str(e)}"
    
    return state

def run_rag(state: ResearchState) -> ResearchState:
    """Run RAG search with enhanced query context"""
    if not state["use_rag"]:
        return state
    
    try:
        logger.info("Running RAG search...")
        enhanced_query = state["query"]
        
        # Add temporal context if available
        if state.get("year"):
            enhanced_query = f"{enhanced_query} (Year: {state['year']})"
        if state.get("quarter"):
            enhanced_query = f"{enhanced_query} (Q{state['quarter']})"
        
        state["rag_result"] = query_pinecone_with_gpt(enhanced_query)
        
    except Exception as e:
        logger.error(f"RAG search failed: {str(e)}", exc_info=True)
        state["error"] = f"RAG error: {str(e)}"
    
    return state

def web_search_agent(state: ResearchState) -> ResearchState:
    """Fetch real-time NVIDIA news with error handling"""
    if not state["use_web_search"]:
        return state
    
    try:
        logger.info("Running web search...")
        state["web_results"] = fetch_nvidia_news(state["query"])
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}", exc_info=True)
        state["error"] = f"Web search error: {str(e)}"
    
    return state

def combine_results(state: ResearchState) -> ResearchState:
    """Combine results from all active sources with proper formatting"""
    if state.get("error"):
        state["final_report"] = f"Error in processing: {state['error']}"
        return state
    
    report_parts = ["## Comprehensive Research Report\n"]
    
    # 1. Snowflake Financial Data (if available)
    if state.get("snowflake_result"):
        sf_result = state["snowflake_result"]
        report_parts.append("### 📈 Financial Data\n")
        report_parts.append(sf_result["text"])
        if sf_result.get("chart_path"):
            report_parts.append(f"\n![Market Cap Chart]({sf_result['chart_path']})")
        report_parts.append("\n")
    
    # 2. RAG Historical Context (if available)
    if state.get("rag_result"):
        report_parts.append("### 📚 Historical Context\n")
        report_parts.append(state["rag_result"])
        report_parts.append("\n")
    
    # 3. Web Search Results (if available)
    if state.get("web_results"):
        report_parts.append("### 🌐 Latest News\n")
        report_parts.append(state["web_results"])
    
    if len(report_parts) == 1:  # Only has the header
        report_parts.append("No relevant information found for your query.")
    
    state["final_report"] = "\n".join(report_parts)
    return state

def error_handler(state: ResearchState) -> ResearchState:
    """Centralized error handling"""
    if state.get("error"):
        state["final_report"] = (
            "⚠️ Research Pipeline Error\n\n"
            f"Error: {state['error']}\n\n"
            "Successful components:\n"
            f"- Snowflake: {'✅' if state.get('snowflake_result') else '❌'}\n"
            f"- RAG: {'✅' if state.get('rag_result') else '❌'}\n"
            f"- Web Search: {'✅' if state.get('web_results') else '❌'}"
        )
    return state

def build_research_pipeline() -> Graph:
    """Build complete research pipeline with all components"""
    graph = Graph()
    
    # Add all nodes
    nodes = [
        ("validate_input", validate_input),
        ("Snowflake", snowflake_agent),
        ("RAG", run_rag),
        ("WebSearch", web_search_agent),
        ("CombineResults", combine_results),
        ("ErrorHandler", error_handler)
    ]
    
    for name, func in nodes:
        graph.add_node(name, func)
    
    # Define execution flow
    def route_after_validation(state: ResearchState) -> str:
        if state.get("error"):
            return "ErrorHandler"
        return "Snowflake" if state["use_snowflake"] else \
               "RAG" if state["use_rag"] else "WebSearch"
    
    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "ErrorHandler": "ErrorHandler",
            "Snowflake": "Snowflake",
            "RAG": "RAG",
            "WebSearch": "WebSearch"
        }
    )
    
    def after_snowflake(state: ResearchState) -> str:
        if state.get("error"):
            return "ErrorHandler"
        return "RAG" if state["use_rag"] else \
               "WebSearch" if state["use_web_search"] else "CombineResults"
    
    graph.add_conditional_edges("Snowflake", after_snowflake)
    
    def after_rag(state: ResearchState) -> str:
        if state.get("error"):
            return "ErrorHandler"
        return "WebSearch" if state["use_web_search"] else "CombineResults"
    
    graph.add_conditional_edges("RAG", after_rag)
    
    graph.add_edge("WebSearch", "CombineResults")
    graph.add_edge("CombineResults", END)
    graph.add_edge("ErrorHandler", END)
    
    graph.set_entry_point("validate_input")
    
    return graph.compile()


if __name__ == "__main__":
    print("=== TESTING INDIVIDUAL COMPONENTS ===")
   
    '''
    print("\n🔍 Testing Web Search Agent (Latest GPU News)")
    web_state = web_search_agent({
        "query": "Latest NVIDIA GPU announcements",
        "use_snowflake": False,
        "use_rag": False,
        "use_web_search": True
    })
    print(web_state["web_results"])
   
    # 1. Test Snowflake
    print("\n🔍 Testing Snowflake Agent (2024 Quarterly Summary)")
    snowflake_state = snowflake_agent({
        "query": "Provide summary of 2024 with the quarters",
        "use_snowflake": True,
        "use_rag": False,
        "use_web_search": False,
    })
    print(snowflake_state["snowflake_result"]["text"])
    
    # 2. Test RAG
    print("\n🔍 Testing RAG Agent (2023 Q2 Product Releases)")
    rag_state = run_rag({
        "query": "What were NVIDIA's major product releases in 2023 Q2?",
        "use_snowflake": False,
        "use_rag": True,
        "use_web_search": False,
        "year": 2023,
        "quarter": 2
    })
    print(rag_state["rag_result"])
    '''
