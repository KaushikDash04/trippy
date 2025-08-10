import os
from typing import List, TypedDict, Annotated
import operator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError
from dotenv import load_dotenv
from serpapi import GoogleSearch # <-- CORRECTED IMPORT

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Define Tools ---
@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a specified city."""
    print(f"--- Calling Weather Tool for {city} ---")
    params = {
        "engine": "google",
        "q": f"weather in {city}",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }
    
    # Correct Usage: Initialize and get results directly
    client = GoogleSearch(params)
    results = client.get_dict() # <-- CORRECTED LINE

    weather_data = results.get("answer_box", {})
    if weather_data and 'weather' in weather_data:
        return f"The weather in {weather_data.get('location', city)} is {weather_data.get('weather')} with a temperature of {weather_data.get('temperature')}."
    elif "weather_results" in results:
        weather_results = results.get("weather_results", {})
        return f"The weather in {weather_results.get('location', city)} is {weather_results.get('condition')} with a temperature of {weather_results.get('temperature')}."
    
    return f"Sorry, I couldn't find the weather for {city}."

# --- 3. Define Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []

class ItineraryDay(BaseModel): # <- Changed from LangchainBaseModel
    day: int
    city: str
    theme: str
    activities: List[str]

class TravelPlan(BaseModel): # <- Changed from LangchainBaseModel
    destination: str
    duration_days: int
    itinerary: List[ItineraryDay]
    weather_forecast: str = "Not available."

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: TravelPlan
    tools: list

# --- 4. Define Agent Nodes ---
# --- 4. Define Agent Nodes ---

def planner_agent_node(state: AgentState):
    """
    A single agent that creates or revises the travel plan based on conversation history.
    - If no plan exists, it creates one.
    - If a plan exists, it modifies it based on the user's last message.
    """
    print("--- Calling Unified Planner/Reviser Agent ---")

    # The prompt is now conditioned on whether a plan already exists.
    if state.get("plan") and state["plan"].itinerary:
        prompt_template = """You are a travel plan revision expert.
            A user wants to modify their existing travel plan based on their latest message.
            Analyze the user's request from the conversation history and update the 'Current Plan' accordingly.
            Keep the parts of the plan that the user did not ask to change.
            Return ONLY the updated TravelPlan JSON object.

            Conversation History:
            {messages}

            Current Plan:
            {plan}
            """
    else:
        prompt_template = """You are a master travel agent.
            Create a detailed, day-by-day itinerary based on the user's request.
            The user's request is in the conversation history.
            - Identify the destination and trip duration.
            - Create a theme for each day.
            - Add 2-4 specific, interesting activities for each day that match the theme.
            - Ensure the activities are logical for the city and daily theme.
            Return ONLY the TravelPlan JSON object.

            Conversation History:
            {messages}

            Current Plan:
            {plan}
            """
    
    # Format the messages for the prompt
    message_content = "\n".join(
        [f"{type(m).__name__}: {m.content}" for m in state["messages"]]
    )
    
    prompt = prompt_template.format(
        messages=message_content, plan=state["plan"].dict() if state.get("plan") else "None"
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_structure = llm.with_structured_output(TravelPlan)
    
    response = model_with_structure.invoke(prompt)
    return {"plan": response, "messages": [AIMessage(content=response.json())]}

def travel_agent_node(state: AgentState):
    print("--- Calling Travel Agent ---")
    prompt = """You are a travel agent. Create a high-level, day-by-day itinerary based on the user's request.
    Identify destination, duration, and interests. Create a theme for each day.
    Do NOT use tools. Return ONLY the TravelPlan JSON object."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_structure = llm.with_structured_output(TravelPlan)
    messages_with_prompt = [HumanMessage(content=prompt)] + state["messages"]
    response = model_with_structure.invoke(messages_with_prompt)
    return {"plan": response}

def interest_agent_node(state: AgentState):
    print("--- Calling Interest Agent ---")
    prompt = f"""You are a travel customization expert for a trip to {state['plan'].destination}.
    Refine the existing plan by adding 2-3 specific activities to each day based on the user's interests from the conversation history.
    Do NOT change the day structure. Return ONLY the updated TravelPlan JSON object.
    Current Plan: {state['plan'].dict()}"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_structure = llm.with_structured_output(TravelPlan)
    messages_with_prompt = [HumanMessage(content=prompt)] + state["messages"]
    response = model_with_structure.invoke(messages_with_prompt)
    return {"plan": response}

# --- RECURSION FIX IS HERE ---
def tool_agent_node(state: AgentState):
    """
    This node now has two jobs:
    1. If it receives a tool result, it updates the plan and finishes.
    2. If not, it decides whether to call a tool.
    """
    print("--- Calling Tool Agent ---")

    # Check if the last message is a ToolMessage (the result from the executor)
    if isinstance(state["messages"][-1], ToolMessage):
        print("--- Processing Tool Result ---")
        plan = state["plan"]
        weather_result = state["messages"][-1].content
        plan.weather_forecast = weather_result
        # The plan is complete. Return the final plan to stop the loop.
        return {"plan": plan}

    # If we are here, it means no tool has been called yet. Decide if one is needed.
    print("--- Deciding to Call Tool ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(state["tools"])
    prompt = f"""You are an assistant. Based on the travel plan, call the `get_weather` tool for the destination city.
    Current Plan: {state['plan'].dict()}"""
    response = llm_with_tools.invoke(prompt)

    # If the LLM makes a tool call, pass it to the executor. Otherwise, the plan is done.
    if response.tool_calls:
        return {"messages": [response]}
    else:
        return {"plan": state["plan"]}

def tool_executor_node(state: AgentState):
    print("--- Executing Tool ---")
    tool_call_message = state["messages"][-1]
    tool_calls = tool_call_message.tool_calls
    tool_results = []
    for call in tool_calls:
        tool_to_call = {t.name: t for t in state["tools"]}[call["name"]]
        result = tool_to_call.invoke(call["args"])
        tool_results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": tool_results}

# --- 5. Define Graph Edges and Conditional Logic ---
def should_continue(state: AgentState) -> str:
    """If the last message has tool calls, continue to execute. Otherwise, end."""
    # This check is now robust. It looks for tool_calls in the last message.
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "continue"
    else:
        return "end"

# --- 6. Construct and Compile the LangGraph ---
graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner_agent", planner_agent_node)
graph_builder.add_node("travel_agent", travel_agent_node)
graph_builder.add_node("interest_agent", interest_agent_node)
graph_builder.add_node("tool_agent", tool_agent_node)
graph_builder.add_node("tool_executor", tool_executor_node)

graph_builder.set_entry_point("planner_agent")
graph_builder.add_edge("planner_agent", "tool_agent")
graph_builder.add_edge("travel_agent", "interest_agent")
graph_builder.add_edge("interest_agent", "tool_agent")
graph_builder.add_conditional_edges(
    "tool_agent",
    should_continue,
    {"continue": "tool_executor", "end": END},
)
graph_builder.add_edge("tool_executor", "tool_agent")
app_graph = graph_builder.compile()

# --- 7. Setup FastAPI Application ---
app = FastAPI(title="AI Trip Planner Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for GraphRecursionError
@app.exception_handler(GraphRecursionError)
async def recursion_error_handler(request: Request, exc: GraphRecursionError):
    return JSONResponse(
        status_code=500,
        content={"message": f"An internal loop error occurred: {exc}"},
    )

@app.post("/plan-trip", response_model=TravelPlan)
def plan_trip(request: ChatRequest):
    history_messages = [
        msg for pair in request.history for msg in [HumanMessage(content=pair[0]), AIMessage(content=str(pair[1]))]
    ]
    history_messages.append(HumanMessage(content=request.message))

    initial_state = {
        "messages": history_messages,
        "plan": None,
        "tools": [get_weather]
    }
    
    final_state = app_graph.invoke(initial_state, {"recursion_limit": 15})
    return final_state["plan"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)