# /chat_client.py

import requests
import json

API_URL = "http://127.0.0.1:8000/plan-trip"

def print_plan(plan_json):
    """Formats and prints the travel plan in a readable way."""
    print("\n--- ✈️ Your Travel Plan ---")
    print(f"Destination: {plan_json['destination']}")
    print(f"Duration: {plan_json['duration_days']} days")
    print("-" * 20)
    for day in plan_json['itinerary']:
        print(f"Day {day['day']} ({day['city']}): {day['theme']}")
        for activity in day['activities']:
            print(f"  - {activity}")
    print("-" * 20)
    print(f"Weather: {plan_json['weather_forecast']}")
    print("--- End of Plan ---\n")

def run_chat():
    """Runs an interactive chat session in the terminal."""
    chat_history = []
    print("Welcome to the AI Trip Planner! Type 'quit' to exit.")

    while True:
        try:
            user_message = input("> ")
            if user_message.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            payload = {"message": user_message, "history": chat_history}
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            ai_response_plan = response.json()
            print_plan(ai_response_plan)
            
            chat_history.append([user_message, json.dumps(ai_response_plan)])

        except requests.exceptions.RequestException as e:
            print(f"\n[Error] Could not connect to the planner API: {e}")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_chat()