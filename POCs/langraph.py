import langgraph
from langgraph.graph import StateGraph
from typing import TypedDict
from websearch import fetch_nvidia_news

# Define the state structure for LangGraph
class ResearchState(TypedDict):
    query: str
    web_results: str

# Web Search function for LangGraph
def web_search_agent(state: ResearchState) -> ResearchState:
    """LangGraph node to fetch real-time NVIDIA news."""
    query = state["query"]
    news_results = fetch_nvidia_news(query)
    return {"web_results": news_results}

# Create a LangGraph pipeline
workflow = StateGraph(ResearchState)

# Add the Web Search Agent as a node
workflow.add_node("web_search", web_search_agent)

# Set entry point
workflow.set_entry_point("web_search")

# Compile the graph
graph = workflow.compile()

# Run the Web Search Agent
result = graph.invoke({"query": "NVIDIA AI market trends"})
print(result["web_results"])
