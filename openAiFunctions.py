from openai import OpenAI, ChatCompletion
import json

client = OpenAI()
llm_model = "gpt-3.5-turbo"


# Example dummy function hard coded to return the same weather
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def call_llm(messages, functions: list = None, function_call: str | dict = "auto") -> ChatCompletion:
    """
    Call open AI
    :param functions: list of functions
    :param messages: Message to be used as prompt
    :param function_call: None, auto,  {"name": "function name"} to force LLM to use this function
    :return: Result from the call to OpenAI as a ChatCompletion object
    """
    if functions is None:
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages
        )
        return response

    # Note the functions parameter used to pass list of functionss that LLM can use
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        functions=functions,
        function_call=function_call
    )
    return response


def app_run():
    """
    Run the different application
    :return:
    """

    # define a function
    # Note that the llm will use the description  listed below to determine whether to use this function
    # Same with the parameters, the llm will use the description to determine whether to use them
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston?"
        }
    ]

    # Call LLM. Let LLM decide if it wants to use the function or not
    response = call_llm(messages, functions, "auto")

    print("------  Call function --------")
    print(f"Response: {response}")
    res_msg = response.choices[0].message
    print(f"Response message: {res_msg}")
    print(f"Response content: {res_msg.content}")
    print(f"Response function to be called: {res_msg.function_call}")
    print(f"Response arguments for function call: {res_msg.function_call.arguments}")
    args = json.loads(res_msg.function_call.arguments)

    # Run function with args returned from llm
    func_result = get_current_weather(args)
    print(func_result)

    # Send results of function call back to llm
    print("------  Send data back to LLM and get response--------")
    messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": func_result,
        }
    )
    response = call_llm(messages)
    print(f"Response content: {response.choices[0].message.content}")

    # Call LLM. Force LLM to use the function
    response = call_llm(messages, functions, {"name": "get_current_weather"})
    res_msg = response.choices[0].message
    print("------  Force call function --------")
    print(f"function args returned: {json.loads(res_msg.function_call.arguments)}")


app_run()
