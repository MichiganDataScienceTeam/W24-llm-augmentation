"""
A simple example of function calling with OpenAI.
We use a get_current_weather function to demonstrate the OpenAI API's ability to call external functions that we define.
It is easy to imagine how much more powerful this could be with a variety of different functions.

Note: This script is simply for learning and demonstration purposes. In practice, you would not hard-code 
function-calling, as libraries like LangChain and LlamaIndex are designed to make this process much easier.

Run with: 
python3 weather-llm.py
"""

import openai
from dotenv import load_dotenv
import os
import requests

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# A really naive implementation of a function that gets the current weather
# In reality, you should be using some weather API to get real-time data
def get_current_weather(location):
    # URL for geocoding API (convert location to geo-coordinates)
    location_api_url = "https://nominatim.openstreetmap.org/search"
    
    # Parameters for the API request
    params = {
        'q': location,
        'format': 'json',
        'limit': 1
    }

    # Send the GET request
    response = requests.get(location_api_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()
        if results:
            # Extract latitude and longitude
            latitude = results[0]['lat']
            longitude = results[0]['lon']
        else:
            return "Location not found"
    else:
        return "Failed to retrieve data"

    # Get the weather API URL for the stations at that location
    weather_station_url = f"https://api.weather.gov/points/{latitude},{longitude}"
    response = requests.get(weather_station_url, headers={"User-Agent": "weatherApp"})
    station_url = response.json()["properties"]["observationStations"]

    # Get the list of stations at that location, and store the first station in the list
    station_list = requests.get(station_url, headers={"User-Agent": "weatherApp"}).json()
    first_station_id = station_list["observationStations"][0].split("/")[-1]

    # Get the latest weather observation at that station
    latest_observation_url = f"https://api.weather.gov/stations/{first_station_id}/observations/latest"
    observation_response = requests.get(latest_observation_url, headers={"User-Agent": "weatherApp"}).json()

    # Return a json string containing all of the oberved weather information
    return str(observation_response)
    

# Function to call the OpenAI API and get a response, whether it's a completion or a tool call
def get_completion(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=300, tools=None):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools
    )
    return response.choices[0].message

if __name__ == "__main__":
    # Define the tools that we want to use
    # This is defined in JSON format for the OpenAI API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get a json string with the current weather conditions at a given location. MUST pass in the input as CITY, STATE. e.g. Mountain View, CA.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },   
        }
    ]

    # Pre-load a message to begin the conversation
    msg = "What's the weather like in Ann Arbor, and what's the air pressure and wind speed there? I want to go on a walk outside, can you suggest an outfit for me?"
    messages = [
        {
            "role": "user",
            "content": msg
        }
    ]
    print(f"User: {msg}\n---")


    response = get_completion(messages, tools=tools)

    # Now, we need to parse the response - the response will contain a TOOL CALL, rather than a completion.
    # The TOOL CALL tells us the function (and appropriate arguments) that the LLM wants to call.
    # This works because OpenAI's API LLMs have been fine-tuned to understand and call functions - other LLMs, such as Llama, do not have this capability.

    # Uncomment the following line to see the response object
    # print(response)
    function_name = response.tool_calls[0].function.name
    function_args = eval(response.tool_calls[0].function.arguments)

    function_call = function_name + "(\'" + (function_args)["location"] + "\')"
    print(f"Calling function: {function_call}\n---")
    tool_response = eval(function_call)
    #print(f"Function returns: {tool_response}")

    messages = [
        {
            "role": "user",
            "content": msg
        },
        {
            "role": "system",
            "content": function_call + " -> " + tool_response,
        }
    ]

    response = get_completion(messages, tools=tools)

    print(f"AI:\n{response.content}\n---")
