# main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, AIMessage
from copilotkit import Action as CopilotAction
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, Action

from schemas import ChatRequest, TravelPlan
from tools.weather_tool import get_weather
from graph import app_graph

# Load environment variables from .env file
load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI(title="AI Trip Planner Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your backend function that the Copilot action will call
async def run_travel_agent(prompt: str):
    """
    Runs your LangGraph agent and returns the travel plan.
    """
    try:
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "plan": None,
            "tools": [get_weather],  # Ensure your tool is included here
        }

        final_state = app_graph.invoke(initial_state, {"recursion_limit": 15})

        # Return dict representation of the plan Pydantic model
        return final_state["plan"].dict()
    except Exception as e:
        print(f"An error occurred in the agent: {e}")
        return {"error": f"Sorry, an error occurred: {e}"}

# Define Copilot action
travel_action = CopilotAction(
    name="displayTravelPlan",
    description="Generates and displays a travel plan based on the user's request.",
    parameters=[
        {
            "name": "prompt",
            "type": "string",
            "description": "The user's request for a travel plan.",
            "required": True,
        }
    ],
    handler=run_travel_agent,
)

# Initialize the CopilotKit SDK with your action
sdk = CopilotKitRemoteEndpoint(actions=[travel_action])

# Add CopilotKit endpoint to your FastAPI app
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")


@app.exception_handler(GraphRecursionError)
async def recursion_error_handler(request: Request, exc: GraphRecursionError):
    return JSONResponse(status_code=500, content={"message": f"An internal loop error occurred: {exc}"})

@app.post("/plan-trip", response_model=TravelPlan)
async def plan_trip(request: ChatRequest):
    """Endpoint to handle chat messages and generate/revise travel plans."""
    history_messages = []
    for pair in request.history:
        history_messages.append(HumanMessage(content=pair[0]))
        history_messages.append(AIMessage(content=str(pair[1])))

    history_messages.append(HumanMessage(content=request.message))

    initial_state = {
        "messages": history_messages,
        "plan": None,
        "tools": [get_weather],
    }

    # Try to find the last plan from history for revision
    for msg in reversed(history_messages):
        if isinstance(msg, AIMessage) and '"itinerary"' in msg.content:
            try:
                last_plan = TravelPlan.parse_raw(msg.content)
                initial_state["plan"] = last_plan
                break
            except Exception:
                continue

    final_state = app_graph.invoke(initial_state, {"recursion_limit": 15})
    return final_state["plan"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
