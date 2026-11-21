import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from harness.tools import registry
from harness.config import (
    MODEL,
    STEP_BUDGET,
    MAX_TOKENS,
    COMPACTION_THRESHOLD,
    COMPACTION_KEEP_RECENT
)
from harness.context import Compactor
from harness.sandbox import Sandbox
from harness.tools.bash import set_sandbox
from harness.memory import (
    load_agents_md,
    save_agents_md,
    validate_agents_md_structure
)

load_dotenv()

client = OpenAI(
    base_url="https://floodgate.g.apple.com/api/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# The synthetic system message injected when the step budget is exceeded.
# It tells the model why it's being asked to stop and what shape its
# response should take.
BUDGET_HIT_MESSAGE = """\
You've reached the step budget for this turn (25 tool calls). Do not make
any more tool calls. Instead, respond directly to the user with:

1. What you accomplished in this turn.
2. What remains to be done.
3. What the user should ask next to continue the work.

Your response will be the final message for this turn. The user will
reply to it and you can continue from there.
"""

# Tool result returned for every tool call in a response that was cut off
# by the max_tokens limit. The call's arguments are incomplete, so we do
# not execute it — we tell the model why and how to recover.
TRUNCATED_CALL_MESSAGE = (
    "error: your response hit the output token limit and was truncated, so "
    "this tool call's arguments are incomplete and it was NOT executed. "
    "Do not retry the same large call. Split the work: call write() with "
    "the first section of the file only, then call append() with each "
    "remaining section (keep each call well under ~2000 words)."
)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text()

# Consolidation prompt loaded from file, like the system prompt.
SESSION_END_MEMORY_PROMPT = (
    Path(__file__).parent / "prompts" / "session_end_memory.txt"
).read_text()

# Cap on how much of each tool result goes into the consolidation
# transcript, so huge outputs (file dumps, logs) don't blow the context.
_TRANSCRIPT_TOOL_RESULT_LIMIT = 2000


def _parse_arguments(raw: str | None) -> dict | None:
    """Parse tool-call arguments JSON. Returns None if it is malformed."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _render_transcript(messages: list) -> str:
    """Flatten the session history into a plain-text transcript.

    The history holds a mix of dicts and ChatCompletionMessage objects,
    including assistant tool calls and tool results. Sending those as-is
    without `tools` is rejected by Bedrock (toolUse/toolResult blocks
    require a toolConfig), and `tool_choice="none"` is not supported by
    the gateway. Rendering everything as text sidesteps both problems.
    """
    lines = []
    for m in messages:
        # Normalise to a dict — assistant tool-call messages are SDK objects.
        if not isinstance(m, dict):
            m = m.model_dump()

        role = m.get("role")
        content = m.get("content") or ""

        # System messages (harness prompt, old AGENTS.md, budget notices)
        # are not part of what happened in the session.
        if role == "system":
            continue

        if role == "user":
            lines.append(f"USER: {content}")
        elif role == "assistant":
            if content:
                lines.append(f"ASSISTANT: {content}")
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {})
                lines.append(
                    f"ASSISTANT TOOL CALL: {fn.get('name')}({fn.get('arguments')})"
                )
        elif role == "tool":
            text = str(content)
            if len(text) > _TRANSCRIPT_TOOL_RESULT_LIMIT:
                text = text[:_TRANSCRIPT_TOOL_RESULT_LIMIT] + "\n...[truncated]"
            lines.append(f"TOOL RESULT: {text}")

    return "\n\n".join(lines)


def _consolidate_memory(messages: list, client) -> None:
    """Run the end-of-session memory consolidation step.

    Sends the current AGENTS.md + full session history + a consolidation
    prompt to the model. Expects a full AGENTS.md rewrite back. Validates
    the structure before writing. On failure, logs and keeps the old file.
    """
    print("Consolidating memory...")

    try:
        # Step 1: gather the inputs — current AGENTS.md and the full history.
        current_agents_md = load_agents_md()

        # Step 2: build the consolidation payload as a single user message
        # containing the prompt, the current AGENTS.md, and a text transcript
        # of the session. Constraints from the Bedrock-backed gateway:
        #   - The conversation must end with a user message (no assistant
        #     prefill), and system messages get hoisted out, so the request
        #     must be user-terminated.
        #   - tool_choice="none" is not supported.
        #   - Raw tool-call / tool-result messages require `tools` to be sent.
        # A flat transcript with no tools satisfies all three and guarantees
        # a text-only response.
        transcript = _render_transcript(messages)
        consolidation_context = (
            f"{SESSION_END_MEMORY_PROMPT}\n\n"
            f"=== Current AGENTS.md ===\n{current_agents_md}\n\n"
            f"=== Session transcript ===\n{transcript}\n"
        )
        consolidation_messages = [
            {"role": "user", "content": consolidation_context},
        ]

        # Step 3: one model call, no tools, so the reply is the file content.
        response = client.chat.completions.create(
            model=MODEL,
            messages=consolidation_messages,
        )
        proposed = response.choices[0].message.content

        if not proposed:
            print("Memory consolidation returned empty content. Keeping current AGENTS.md.")
            return

        # Step 4: validate structure. Reject a malformed return without
        # touching the file.
        if not validate_agents_md_structure(proposed):
            print(
                "Memory consolidation returned malformed structure "
                "(missing section headers). Keeping current AGENTS.md."
            )
            return

        # Step 5: atomic write. The old file is replaced by the new one
        # in a single filesystem operation.
        save_agents_md(proposed)
        print("Memory updated.")

    except Exception as e:
        # If anything goes wrong during consolidation — API error, timeout,
        # unexpected exception — we log and return. The old AGENTS.md stays
        # intact. Consolidation is a nice-to-have; sandbox teardown is
        # non-negotiable.
        print(f"Memory consolidation failed: {e}. Keeping current AGENTS.md.")


def _run_compaction(compactor: Compactor, messages: list[dict]) -> None:
    before = compactor.get_last_token_count()
    new_messages = compactor.compact(messages)

    if new_messages is None:
        # No-op: not enough conversation to compact.
        print(

            "[Compaction skipped: not enough older history to summarize "
            f"(need more than {COMPACTION_KEEP_RECENT} user turns).]"
        )
        return

    messages.clear()
    messages.extend(new_messages)

    approximate_after = compactor.approximate_char_count(messages)
    print(
        f"[Context compacted at {before:,} tokens → ~{approximate_after:,} tokens. "
        f"Full history: .harness/context_log.jsonl]"
    )


def run():
    """Run the agent's conversation loop until the user quits."""

    # Step 1: start the sandbox and wire it into the bash tool.            
    # This creates the Docker container, bind-mounts the workspace, and
    # cleans up any orphan containers from prior crashed sessions.
    sandbox = Sandbox()
    sandbox.start()
    set_sandbox(sandbox)

    # Create the Compactor once per session. Its state (last-seen token 
    # count) needs to persist across turns, so it lives outside the loop.
    compactor = Compactor(client)

    try:
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

        print("Agent ready. Type 'quit' or 'exit' to leave. \nType /compact to force compaction.\n")

        while True:
            # Compaction check runs BEFORE reading user input. If the
            # previous turn crossed the threshold, compact now so the
            # next turn starts against a lean context.
            if compactor.should_compact():
                _run_compaction(compactor, messages)

            # Display current context size before the prompt so students 
            # can see themselves approaching the compaction threshold.
            # Suppressed on the first turn (no model call has happened
            # yet, so the count is zero and meaningless).
            current_tokens = compactor.get_last_token_count()
            if current_tokens > 0:
                print(f"[Context: {current_tokens:,} / {COMPACTION_THRESHOLD:,} tokens]")

            # 1. Get input from the user
            user_input = input("you > ").strip()

            # 2. Allow the user to leave cleanly
            if user_input in {"quit", "exit"}:
                _consolidate_memory(messages, client)
                print("Goodbye.")
                break

            # skip empty lines without making a model call
            if not user_input:
                continue

            if user_input == "/compact":
                # Manual compaction — same code path as automatic, but
                # doesn't wait for the threshold. Useful for demonstrations
                # and for the user to trigger cleanup when they know the
                # context is heavy.
                _run_compaction(compactor, messages)
                continue

            # 3. Append the user's message to the history
            messages.append({"role": "user", "content": user_input})

            # 4. Full ReAct dispatch loop — replaces the single-round dispatch
            step_count = 0
            while True:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=registry.get_schemas(),
                    max_tokens=MAX_TOKENS,
                )

                if response.usage:
                    compactor.record_token_usage(response.usage.prompt_tokens)

                choice = response.choices[0]
                message = choice.message
                truncated = choice.finish_reason == "length"

                if truncated and not message.tool_calls:
                    # Plain-text reply cut off. Show what we got, flagged.
                    print("[warning] response hit max_tokens and was truncated")

                if not message.tool_calls:
                    break

                if step_count >= STEP_BUDGET:
                    messages.append({"role": "system", "content": BUDGET_HIT_MESSAGE})
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=registry.get_schemas(),
                        max_tokens=MAX_TOKENS,
                        # No tool_choice: "none" is not supported by the
                        # Bedrock-backed gateway. BUDGET_HIT_MESSAGE tells the
                        # model not to call tools instead.
                    )

                    if response.usage:
                        compactor.record_token_usage(response.usage.prompt_tokens)

                    message = response.choices[0].message
                    break

                # Parse arguments up front. Truncated or malformed JSON must
                # not be echoed back into history verbatim — the gateway
                # would reject it on the next request — so we sanitise it.
                parsed_calls = []
                for call in message.tool_calls:
                    args = _parse_arguments(call.function.arguments)
                    parsed_calls.append((call, args))

                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": json.dumps(args if args is not None else {}),
                            },
                        }
                        for call, args in parsed_calls
                    ],
                })

                if truncated:
                    print(
                        "[warning] response hit max_tokens; tool call(s) "
                        "truncated and skipped"
                    )

                for call, args in parsed_calls:
                    if truncated:
                        result = TRUNCATED_CALL_MESSAGE
                    elif args is None:
                        result = (
                            "error: tool call arguments were not valid JSON; "
                            "the call was not executed."
                        )
                    else:
                        result = registry.dispatch(call.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })

                step_count += 1

            # 5. After the loop: `message.content` should have real text. If
            # it doesn't, something unexpected happened (rare API edge case or
            # a bug in the loop termination logic). Raise loudly rather than
            # silently substituting a placeholder.
            if not message.content:
                raise RuntimeError(
                    "Loop terminated but message.content is empty. "
                    "This shouldn't happen — check the API response and the "
                    "termination logic."
                )

            # 6. By here, `message` is the model's final text response for this turn
            assistant_text = message.content
            messages.append({"role": "assistant", "content": assistant_text})

            # 7. Show the user
            print(f"\nagent > {assistant_text}\n")

    finally:
        # Step 2: tear down the sandbox no matter how run() exits.
        # This runs on normal exit, on exception, on user Ctrl-C — but
        # NOT on hard crashes (SIGKILL, power loss). The orphan cleanup
        # in Sandbox.start() handles those on the next session start.
        print("Stopping sandbox...")
        sandbox.stop()
        print("Sandbox stopped.")


if __name__ == "__main__":
    run()