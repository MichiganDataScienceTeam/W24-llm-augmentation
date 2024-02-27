"""
A simple example of function calling with OpenAI.
We use a get_current_weather function to demonstrate the OpenAI API's ability to call external functions that we define.
It is easy to imagine how much more powerful this could be with real-world APIs, and with a variety of different functions.

Note: This script is simply for learning and demonstration purposes. In practice, you would use a weather API, and you would not 
hard-code function-calling; libraries like LangChain and LlamaIndex are designed to make this process much easier.

Run with: 
python3 weather-llm.py
"""

import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# A really naive implementation of a function that gets the current weather
# In reality, you should be using some weather API to get real-time data
def get_current_weather(location):
    if location == "Mountain View, CA":
        return "sunny"
    elif location == "Seattle, WA":
        return "rainy"
    else:
        return "unknown"

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
                "description": "Get the current weather in a given location. MUST pass in the input as  CITY, STATE. e.g. Mountain View, CA.",
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
    msg = "What is the weather like in Seattle?"
    messages = [
        {
            "role": "user",
            "content": msg
        }
    ]
    print(f"User: {msg}")


    response = get_completion(messages, tools=tools)

    # Now, we need to parse the response - the response will contain a TOOL CALL, rather than a completion.
    # The TOOL CALL tells us the function (and appropriate arguments) that the LLM wants to call.
    # This works because OpenAI's API LLMs have been fine-tuned to understand and call functions - other LLMs, such as Llama, do not have this capability.

    # Uncomment the following line to see the response object
    # print(response)
    function_name = response.tool_calls[0].function.name
    function_args = eval(response.tool_calls[0].function.arguments)

    function_call = function_name + "(\'" + (function_args)["location"] + "\')"
    print(f"Calling function: {function_call}")
    tool_response = eval(function_call)
    print(f"\tResponse: {tool_response}")

    messages = [
        {
            "role": "user",
            "content": "What is the weather like in Seattle?"
        },
        {
            "role": "system",
            "content": function_call + " -> " + tool_response,
        }
    ]

    response = get_completion(messages, tools=tools)

    print(f"AI: {response.content}")

