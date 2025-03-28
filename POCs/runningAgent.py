from langgraph.graph import Graph
from langgraph.constants import END, START
from typing import Dict, Any, TypedDict, Optional, Literal
import logging

# Importing individual agent functions
from websearch import fetch_nvidia_news
from hybrid_search_pinecone_assign5 import query_pinecone_with_gpt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state structure
class ResearchState(TypedDict):
    query: str
    use_rag: bool
    use_web_search: bool
    year: Optional[int]
    quarter: Optional[str]
    rag_result: Optional[str]
    web_results: Optional[str]
    final_report: Optional[str]
    error: Optional[str]

def validate_input(state: ResearchState) -> ResearchState:
    """Validate input parameters before processing."""
    if not state.get("query"):
        state["error"] = "Query cannot be empty"
    elif not (state["use_rag"] or state["use_web_search"]):
        state["error"] = "At least one search method must be enabled"
    return state

def web_search_agent(state: ResearchState) -> ResearchState:
    """Fetch real-time NVIDIA news with error handling."""
    if state["use_web_search"]:
        try:
            logger.info("Running web search...")
            state["web_results"] = fetch_nvidia_news(state["query"])
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            state["error"] = f"Web search error: {str(e)}"
    return state

def run_rag(state: ResearchState) -> ResearchState:
    """Invoke RAG with metadata filtering and error handling."""
    if state["use_rag"]:
        try:
            logger.info("Running RAG search...")
            # Enhance query with metadata if available
            enhanced_query = state["query"]
            if state.get("year") and state.get("quarter"):
                enhanced_query = f"{enhanced_query} (Filtering for {state['quarter']} {state['year']})"
            
            state["rag_result"] = query_pinecone_with_gpt(enhanced_query)
        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            state["error"] = f"RAG search error: {str(e)}"
    return state

def combine_results(state: ResearchState) -> ResearchState:
    """Combine results from both agents with quality checks."""
    logger.info("Combining results...")
    
    if state.get("error"):
        state["final_report"] = f"Error in processing: {state['error']}"
        return state
    
    report = "## Comprehensive Research Report\n\n"
    
    # Add RAG results if available
    if state.get("rag_result"):
        report += f"### Historical Data Analysis\n{state['rag_result']}\n\n"
    
    # Add web results if available
    if state.get("web_results"):
        report += f"### Latest Market Insights\n{state['web_results']}\n"
    
    # Handle case where no results were found
    if not report.strip():
        report = "No relevant information found for your query."
    
    state["final_report"] = report
    return state

def error_handler(state: ResearchState) -> ResearchState:
    """Handle errors and generate fallback output."""
    if state.get("error"):
        state["final_report"] = f"⚠️ Processing Error: {state['error']}\n\nWe encountered an issue while processing your request."
    return state

def build_research_pipeline() -> Graph:
    """Build and compile the research pipeline graph."""
    graph = Graph()

    # Add nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("RAG", run_rag)
    graph.add_node("WebSearch", web_search_agent)
    graph.add_node("CombineResults", combine_results)
    graph.add_node("ErrorHandler", error_handler)

    # Define conditional edges from validation
    def route_after_validation(state: ResearchState) -> str:
        if state.get("error"):
            return "ErrorHandler"
        if state["use_rag"] and state["use_web_search"]:
            return "both"
        return "rag" if state["use_rag"] else "web"

    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "ErrorHandler": "ErrorHandler",
            "both": "RAG",
            "rag": "RAG",
            "web": "WebSearch"
        }
    )

    # Define conditional edges from RAG
    def after_rag(state: ResearchState) -> str:
        if state["use_web_search"]:
            return "to_web"
        return "to_combine"

    graph.add_conditional_edges(
        "RAG",
        after_rag,
        {
            "to_web": "WebSearch",
            "to_combine": "CombineResults"
        }
    )

    # Normal flows
    graph.add_edge("WebSearch", "CombineResults")
    graph.add_edge("CombineResults", END)
    graph.add_edge("ErrorHandler", END)

    # Set entry point
    graph.set_entry_point("validate_input")

    return graph.compile()
'''
if __name__ == "__main__":
    # Test the pipeline construction first
    try:
        pipeline = build_research_pipeline()
        print("✅ Pipeline built successfully!")
    except Exception as e:
        print(f"❌ Failed to build pipeline: {str(e)}")
        raise

    # Define test cases
    test_cases = [
        {
            "name": "RAG Only - Valid Query",
            "state": {
                "query": "NVIDIA Q3 2023 financial results",
                "use_rag": True,
                "use_web_search": False,
                "year": 2023,
                "quarter": "Q3"
            },
            "expected": {
                "has_report": True,
                "has_rag": True,
                "has_web": False
            }
        },
        {
            "name": "Web Search Only - Valid Query",
            "state": {
                "query": "Latest NVIDIA GPU announcements",
                "use_rag": False,
                "use_web_search": True
            },
            "expected": {
                "has_report": True,
                "has_rag": False,
                "has_web": True
            }
        },
        {
            "name": "Combined RAG + Web - Valid Query",
            "state": {
                "query": "Complete analysis of NVIDIA's performance",
                "use_rag": True,
                "use_web_search": True
            },
            "expected": {
                "has_report": True,
                "has_rag": True,
                "has_web": True
            }
        },
        {
            "name": "Invalid Query - Empty String",
            "state": {
                "query": "",
                "use_rag": True,
                "use_web_search": False
            },
            "expected": {
                "has_error": True,
                "error_contains": "Query cannot be empty"
            }
        },
        {
            "name": "Invalid Configuration - No Sources Selected",
            "state": {
                "query": "NVIDIA stock performance",
                "use_rag": False,
                "use_web_search": False
            },
            "expected": {
                "has_error": True,
                "error_contains": "At least one search method must be enabled"
            }
        }
    ]

    # Run test cases
    for case in test_cases:
        print(f"\n🔍 Running test: {case['name']}")
        print(f"Input state: {case['state']}")
        
        try:
            result = pipeline.invoke(case["state"])
            print("\n=== Pipeline Result ===")
            print(result)
            
            # Validate results
            if 'expected' in case:
                if case['expected'].get('has_error', False):
                    if not result.get('error'):
                        print("❌ Expected error but none was found")
                    elif case['expected'].get('error_contains') and case['expected']['error_contains'] not in result.get('error', ''):
                        print(f"❌ Error message doesn't contain expected text: {case['expected']['error_contains']}")
                    else:
                        print("✅ Error case handled correctly")
                else:
                    if not result.get('final_report'):
                        print("❌ No report generated")
                    else:
                        if case['expected'].get('has_rag', False) and not result.get('rag_result'):
                            print("❌ Expected RAG results but none found")
                        elif not case['expected'].get('has_rag', False) and result.get('rag_result'):
                            print("❌ Unexpected RAG results found")
                        
                        if case['expected'].get('has_web', False) and not result.get('web_results'):
                            print("❌ Expected web results but none found")
                        elif not case['expected'].get('has_web', False) and result.get('web_results'):
                            print("❌ Unexpected web results found")
                        
                        print("✅ Report generated with expected components")
            
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            continue

        print("Test completed")'
        '''