# /tools/weather_tool.py

import os
from langchain.tools import tool
from serpapi import GoogleSearch

@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a specified city."""
    print(f"--- Calling Weather Tool for {city} ---")
    params = {
        "engine": "google",
        "q": f"weather in {city}",
        "api_key": os.getenv("SERPAPI_API_KEY"),
    }
    client = GoogleSearch(params)
    results = client.get_dict()

    weather_data = results.get("answer_box", {})
    if weather_data and 'weather' in weather_data:
        temp = weather_data.get('temperature', '?')
        unit = weather_data.get('unit', 'F') # Default to F if not specified
        return f"The weather in {weather_data.get('location', city)} is {weather_data.get('weather')} with a temperature of {temp}°{unit}."
    elif "weather_results" in results:
        weather_results = results.get("weather_results", {})
        temp = weather_results.get('temperature', '?')
        unit = weather_results.get('unit', 'F')
        return f"The weather in {weather_results.get('location', city)} is {weather_results.get('condition')} with a temperature of {temp}°{unit}."
    
    return f"Sorry, I couldn't find the weather for {city}."