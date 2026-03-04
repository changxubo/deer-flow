from langgraph.graph import StateGraph,START,END
from src.agents.lead_agent.agent import make_lead_agent
from src.agents.thread_state import ThreadState

def create_lead_graph():
    """
    Creates a simple LangGraph graph with a single agent node.
    """
    graph = StateGraph(ThreadState)
    lead_agent = make_lead_agent()
    graph.add_node("agent", lead_agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    runnable = graph.compile()
    return runnable


