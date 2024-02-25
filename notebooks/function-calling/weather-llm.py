import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_current_weather(location):
    if location == "Mountain View, CA":
        return "sunny"
    elif location == "Seattle, WA":
        return "rainy"
    else:
        return "unknown"

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

    messages = [
        {
            "role": "user",
            "content": "What is the weather like in Seattle?"
        }
    ]
    
    print("User: What is the weather like in Seattle?")

    response = get_completion(messages, tools=tools)

    # Parse the response function call
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

