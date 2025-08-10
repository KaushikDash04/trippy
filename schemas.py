# /schemas.py

from typing import List, TypedDict, Annotated
import operator
from pydantic import BaseModel

# --- Pydantic Models for Data Validation ---

class ChatRequest(BaseModel):
    """The structure of a request from the client to the backend."""
    message: str
    history: List[List[str]] = []

class ItineraryDay(BaseModel):
    """A model for a single day in the travel itinerary."""
    day: int
    city: str
    theme: str
    activities: List[str]

class TravelPlan(BaseModel):
    """The main model for the entire travel plan."""
    destination: str
    duration_days: int
    itinerary: List[ItineraryDay]
    weather_forecast: str = "Not available."

# --- TypedDict for Graph State ---

class AgentState(TypedDict):
    """Defines the state that flows through the LangGraph."""
    messages: Annotated[list, operator.add]
    plan: TravelPlan
    tools: list