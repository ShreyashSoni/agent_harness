import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from harness.tools import registry
from harness.memory import load_agents_md

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

You also have six git tools — git_status, git_diff, git_log, git_commit, git_checkout, git_branch — for versioning your work. The workspace is
already initialized as a git repo. Use git to:
- Commit frequently. Small, focused commits are easier to roll back.
- Commit before doing anything risky (large rewrites, deleting files, restructuring). A commit before the risky step gives you a recovery point.
- Write meaningful commit messages — describe what changed and why, in the present tense (e.g., "add user authentication module").
- Branch experiments. When trying an alternative approach, create a branch first so the main line of work stays intact.

The workspace contains an `AGENTS.md` file — your durable memory across    
sessions. It is automatically loaded into your context at the start of
every session. Update it (using the `write` tool) when you learn
something worth remembering for future sessions. Good things to write:
- Project context: what this codebase is, what it does, who uses it
- Conventions you've observed: code style, libraries, naming patterns
- Decisions that have been made and the reasoning behind them
- Gotchas: quirks, non-obvious dependencies, things that have tripped
  up earlier sessions
- Active tasks: what's currently being worked on (clear when complete)

When updating AGENTS.md, preserve the existing structure (the section
headings). Add to the relevant section instead of replacing the whole
file. If the section starts with a parenthetical hint like "(What is
this project?)", replace the hint with real content as you fill it in.
"""


def run():
    """Run the agent's conversation loop until the user quits."""

    # Load AGENTS.md and assemble the initial message list.       
    # The first system message is the harness's prompt; the second is the
    # project's accumulated memory.
    agents_md = load_agents_md()

    # The conversation history. This is the entire memory of the agent.
    # Every turn, we append to it and send the whole thing to the model.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": agents_md}
    ]

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
        assistant_text = message.content or "(no text response — used tools only)"
        messages.append({"role": "assistant", "content": assistant_text})

        # 7. Show the user
        print(f"\nagent > {assistant_text}\n")

    return


if __name__ == "__main__":
    run()