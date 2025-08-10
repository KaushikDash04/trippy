# /agents.py

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from schemas import AgentState, TravelPlan

# --- Agent Node Functions ---

def planner_agent_node(state: AgentState):
    """A single agent that creates or revises the travel plan based on conversation history."""
    print("--- Calling Unified Planner/Reviser Agent ---")

    if state.get("plan") and state["plan"].itinerary:
        prompt_template = """You are a travel plan revision expert. A user wants to modify their existing plan.
Analyze the user's latest message and update the 'Current Plan' accordingly.
Keep the parts of the plan the user did not ask to change. Return ONLY the updated TravelPlan JSON object.
Conversation History:
{messages}
Current Plan:
{plan}"""
    else:
        prompt_template = """You are a master travel agent. Create a detailed, day-by-day itinerary based on the user's request.
The user's request is in the conversation history. Identify destination and duration. Create a theme for each day.
Add 2-4 specific, interesting activities for each day. Return ONLY the TravelPlan JSON object.
Conversation History:
{messages}
Current Plan: {plan}"""
    
    message_content = "\n".join([f"{type(m).__name__}: {m.content}" for m in state["messages"]])
    prompt = prompt_template.format(messages=message_content, plan=state["plan"].dict() if state.get("plan") else "None")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_structure = llm.with_structured_output(TravelPlan)
    response = model_with_structure.invoke(prompt)
    
    return {"plan": response, "messages": [AIMessage(content=response.json())]}

def tool_agent_node(state: AgentState):
    """This node decides whether to call a tool or not."""
    print("--- Calling Tool Agent ---")

    if isinstance(state["messages"][-1], ToolMessage):
        print("--- Processing Tool Result ---")
        plan = state["plan"]
        weather_result = state["messages"][-1].content
        plan.weather_forecast = weather_result
        return {"plan": plan}

    print("--- Deciding to Call Tool ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(state["tools"])
    
    prompt = f"""Based on the travel plan, should you call the `get_weather` tool for the primary destination city?
If the weather forecast is already present and not 'Not available.', you don't need to call it again.
Current Plan: {state['plan'].dict()}"""
    
    response = llm_with_tools.invoke(prompt)

    if response.tool_calls:
        return {"messages": [response]}
    else:
        return {"plan": state["plan"]}

def tool_executor_node(state: AgentState):
    """This node executes the tools that have been called."""
    print("--- Executing Tool ---")
    tool_call_message = state["messages"][-1]
    tool_calls = tool_call_message.tool_calls
    tool_results = []
    
    for call in tool_calls:
        tool_to_call = {t.name: t for t in state["tools"]}[call["name"]]
        result = tool_to_call.invoke(call["args"])
        tool_results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        
    return {"messages": tool_results}