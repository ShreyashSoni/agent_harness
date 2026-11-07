import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from harness.tools import registry

load_dotenv()

MODEL = "anthropic.claude-sonnet-5"

client = OpenAI(
    base_url="https://floodgate.g.apple.com/api/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """
You are a coding assistant running in a terminal, helping a developer with software engineering tasks.

Be concise. Prefer short, direct answers over long ones. When the user asks for code, return the code with minimal explanation unless they ask for more.

When returning code, use fenced code blocks and specify the language.

You have access to five filesystem tools — read, write, list, mkdir, delete — operating on a workspace directory. Use them whenever a task involves reading, modifying, or organizing files. Paths are relative to
the workspace root. Prefer reading and writing real files over describing them in conversation.
"""

def run():
    """Run the agent's conversation loop until the user quits."""

    # The conversation history. This is the entire memory of the agent.
    # Every turn, we append to it and send the whole thing to the model.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Agent ready. Type 'quit' or 'exit' to leave.\n")

    while True:

        # 1. Get input from the user
        user_input = input("you > ").strip()

        # 2. Allow the user to leave cleanly
        if user_input in {"quit", "exit"}:
            print("Goodbye.")
            break

        # skip empty lines without making a model call
        if not user_input:
            continue

        # 3. Append the user's message to the history
        messages.append({
            "role": "user",
            "content": user_input
        })

        # 4. Call the model with the full conversation so far and the available tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=registry.get_schemas()
        )

        # 5. Extract the assistant's reply
        message = response.choices[0].message
        # If the model asked for a tool call, handle it before producing    
        # the user-facing reply. Minimum-viable dispatch: one round only.
        if message.tool_calls:
            # Step 1: record the model's tool-call message in history so the
            # upcoming tool-result messages have something to reference.
            messages.append(message)
            # Step 2: run each requested tool and append its result to history,
            # using the matching tool_call_id so the model can pair them up.
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                result = registry.dispatch(call.function.name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })
            # Step 3: re-call the model now that the tool results are in
            # context. This second call produces the model's final text reply.
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=registry.get_schemas()
            )
            message = response.choices[0].message

        # 6. By here, `message` is the model's final text response for this turn —
        # either from the first call (no tools needed) or the second (after dispatch).
        assistant_text = message.content
        messages.append({"role": "assistant", "content": assistant_text})

        # 7. Show the user
        print(f"\nagent > {assistant_text}\n")

    return


if __name__ == "__main__":
    run()