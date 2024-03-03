"""
A simple example of function calling with OpenAI.
We use a get_current_weather function to demonstrate the OpenAI API's ability to call external functions that we define.
It is easy to imagine how much more powerful this could be with a variety of different functions.

Note: This script is simply for learning and demonstration purposes. 
In practice, you would not "hard-code" LLM conversation as we are. 
Libraries like LangChain and LlamaIndex are designed to make this process much easier.

Run with: 
python3 weather-llm.py
"""

import openai
from dotenv import load_dotenv
import os
import requests

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# A function that gets weather data given a location in a string format (e.g., "Ann Arbor, MI", "Seattle, WA", ...)
# First, the location is geocoded into a (latitude, longitude) coordinate
# Then, using the coordinate, real-time weather data is fetched
def get_current_weather(location: str) -> str:
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
            raise Exception(f"Location {location} not found")
    else:
        raise Exception("Failed to retrieve data")

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

    # Return a string containing all of the oberved weather information
    # Uncomment the following line to see the returned object
    # print(str(observation_response))

    # We could just return the entire json object as a string, but that is very unreliable with LLMs (think about tokenization)
    # A better practice would be to extract data into a more "human" form, and return that
    return f"""
    Weather Data for {location}:

    Current Weather: {observation_response['properties']['textDescription']}
    Current Temperature: {observation_response['properties']['temperature']['value']} degrees celsius
    Windspeed: {observation_response['properties']['windSpeed']['value']} km per hr
    Visibility: {observation_response['properties']['visibility']['value']} meters
    Wind Chill: {observation_response['properties']['windChill']['value']} degrees celsius
    Relative Humidity: {observation_response['properties']['relativeHumidity']['value']} %
    """
    

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

    tool_responses = []
    # multiple functions can be called at the same time, so we must account for that
    for tool_call in response.tool_calls:
        function_name = tool_call.function.name
        function_args = eval(tool_call.function.arguments)

        function_call = function_name + "(\'" + (function_args)["location"] + "\')"
        print(f"Calling function: {function_call}")
        tool_response = eval(function_call)
        print(f"Function returns: {tool_response}\n---")
        tool_responses.append({"function_name": function_name, "tool_response": tool_response})

    # Adjust the messages array for next API call
    messages = [
        {
            "role": "user",
            "content": msg
        },
        response,
    ]

    for idx, tool_call in enumerate(response.tool_calls):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_responses[idx]['function_name'],
                "content": tool_responses[idx]['tool_response'],
            }
        )
        

    response = get_completion(messages, tools=tools)

    print(f"AI: {response.content}\n---")
