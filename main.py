import json
from openai import OpenAI
from omegaconf import OmegaConf

# Read config
config = OmegaConf.load("config.yaml")

# Setup openai API client
client = OpenAI(
    base_url=config.api_host,
    api_key=config.api_key
)


# Create tool (read file content)
def read_file(filename: str):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


# Define tool specification for LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a specific local text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read, e.g., 'secrets.txt'",
                    }
                },
                "required": ["filename"],
            },
        },
    }
]

user_prompt = """
I've hidden some treasure that's for you to find. You can starting by looking at the clues in treasuremap.txt
"""

print(f"Starting prompt: {user_prompt}")

# STEP 1: Send the prompt and the tool definition to the LLM
response = client.chat.completions.create(
    model=config.api_model,  # Or the specific model name used at your lab
    messages=[{"role": "user", "content": user_prompt}],
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
messages = [
    {"role": "user", "content": user_prompt}
]

while tool_calls is not None:
    # Add response to message history
    messages.append(response_message)

    print(f"LLM wants to call {len(tool_calls)} tool(s).")

    # Execute requested tool calls
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        match function_name:
            case "read_file":
                print(f"Executing local function: read_file('{function_args['filename']}')")
                observation = read_file(function_args['filename'])

                # Add the result to history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": observation,
                })
            case _:
                print(f"Error: function '{function_name}' not found.")

    # Get new response given tool call output
    response = client.chat.completions.create(
        model=config.api_model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

print(f"\nFinal LLM Answer:\n'{response.choices[0].message.content}'")
