# /graph.py

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from schemas import AgentState
from agents import planner_agent_node, tool_agent_node, tool_executor_node

# --- Graph Definition ---

def should_continue(state: AgentState) -> str:
    """If the last message has tool calls, continue to execute. Otherwise, end."""
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "continue"
    return "end"

# Create the state graph
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("planner_agent", planner_agent_node)
graph_builder.add_node("tool_agent", tool_agent_node)
graph_builder.add_node("tool_executor", tool_executor_node)

# Define the graph's flow
graph_builder.set_entry_point("planner_agent")
graph_builder.add_edge("planner_agent", "tool_agent")
graph_builder.add_conditional_edges(
    "tool_agent",
    should_continue,
    {"continue": "tool_executor", "end": END},
)
graph_builder.add_edge("tool_executor", "tool_agent")

# Compile the graph
app_graph = graph_builder.compile()