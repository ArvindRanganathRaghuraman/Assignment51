from langgraph.graph import Graph
from langgraph.constants import END, START
from typing import Dict, Any, TypedDict, Optional

# Importing individual agent functions
from websearch import fetch_nvidia_news
from hybrid_search_pinecone_assign5 import query_pinecone_with_gpt

# Define state structure
class ResearchState(TypedDict):
    query: str
    use_rag: bool
    use_web_search: bool
    year: Optional[int]
    quarter: Optional[str]
    rag_result: str | None
    web_results: str | None
    final_report: str | None

# Define agent functions
def web_search_agent(state: ResearchState) -> ResearchState:
    """Fetch real-time NVIDIA news."""
    if state["use_web_search"]:
        print("Running web search...")
        state["web_results"] = fetch_nvidia_news(state["query"])
    return state

def run_rag(state: ResearchState) -> ResearchState:
    """Invoke the RAG agent using Pinecone hybrid search."""
    if state["use_rag"]:
        print("Running RAG search...")
        state["rag_result"] = query_pinecone_with_gpt(state["query"])
    return state

def combine_results(state: ResearchState) -> ResearchState:
    """Combine results from both agents."""
    print("Combining results...")
    report = "Comprehensive Research Report:\n"
    
    # Safely access results
    if state.get("rag_result"):
        report += f"\n### Historical Data (RAG):\n{state['rag_result']}\n"
    if state.get("web_results"):
        report += f"\n### Real-Time Insights (Web Search):\n{state['web_results']}\n"
    
    state["final_report"] = report
    return state

def build_file():
    # Create the pipeline
    graph = Graph()

    # Add nodes (agents)
    graph.add_node("RAG", run_rag)
    graph.add_node("WebSearch", web_search_agent)
    graph.add_node("CombineResults", combine_results)

    # Define edges between nodes (workflow)
    graph.add_edge("RAG", "CombineResults")
    graph.add_edge("WebSearch", "CombineResults")
    graph.add_edge("CombineResults", END)

    # Define entry point based on conditions
    def router(state: ResearchState) -> str:
        if state.get("use_rag", False) and state.get("use_web_search", False):
            return "both"
        elif state.get("use_rag", False):
            return "rag"
        elif state.get("use_web_search", False):
            return "web"
        else:
            raise ValueError("At least one of use_rag or use_web_search must be True")

    # Add conditional edges from START
    graph.add_conditional_edges(
        START,
        router,
        {
            "rag": "RAG",
            "web": "WebSearch",
            "both": "RAG"  # Start with RAG, WebSearch will run automatically
        }
    )

    return graph.compile()
'''
# Self-testing
if __name__ == "__main__":
    try:
        pipeline = build_file()
        print("Pipeline built successfully!")
        
        # Test cases
        test_cases = [
            {
                "name": "RAG Only",
                "state": {
                    "query": "NVIDIA Q4 2023 financial results",
                    "use_rag": True,
                    "use_web_search": False
                }
            },
            {
                "name": "Web Only",
                "state": {
                    "query": "Latest NVIDIA news",
                    "use_rag": False,
                    "use_web_search": True
                }
            },
            {
                "name": "Both",
                "state": {
                    "query": "Complete NVIDIA Q3 2023 analysis",
                    "use_rag": True,
                    "use_web_search": True
                }
            }
        ]
        
        for case in test_cases:
            print(f"\nRunning test: {case['name']}")
            result = pipeline.invoke(case["state"])
            if 'final_report' in result:
                print(f"Final report preview:\n{result['final_report'][:200]}...")
            else:
                print("No final report generated")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise '''