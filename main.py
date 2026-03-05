from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.config import AgentConfig
from agent.graph import build_init_graph, build_turn_graph
from agent.state import BuildState, CodebaseIndex


def main() -> None:
    """Interactive REPL — the user asks questions, the agent responds."""
    load_dotenv()

    config = AgentConfig()

    # Verify workspace exists
    ws = config.workspace_path
    if not ws.exists() or not ws.is_dir():
        print(f"ERROR: workspace not found at '{ws.resolve()}'")
        print("The agent requires a pre-downloaded copy of nlohmann/json at workspace/json/")
        sys.exit(1)

    print("=" * 60)
    print("  nlohmann/json Codebase Assistant")
    print(f"  Workspace: {ws.resolve()}")
    print(f"  Model: {config.model_name}")
    print(f"  Token budget: {config.token_budget}")
    print("=" * 60)
    print("Commands: ask questions, 'build'/'compile', 'test'/'run', 'exit'/'quit'")
    print()

    # Phase 1: Index the workspace (one-shot)
    print("Indexing workspace...")
    init_graph = build_init_graph(config)
    init_state = {
        "messages": [],
        "summary_of_knowledge": "",
        "codebase_index": CodebaseIndex(),
        "current_intent": "QUESTION",
        "build_state": BuildState(),
        "turn_count": 0,
        "last_user_input": "",
        "_retrieved_context": "",
        "_tool_iteration_count": 0,
        "_turn_subagent_count": 0,
        "_turn_debug_logs": [],
    }

    try:
        state = init_graph.invoke(init_state, config={"recursion_limit": 10})
        summary = state.get("summary_of_knowledge", "")
        print(f"Ready! {summary[:200]}")
    except Exception as e:
        print(f"Startup error: {e}")
        state = init_state

    print()

    # Phase 2: Build per-turn graph
    turn_graph = build_turn_graph(config)

    # REPL loop
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Check for exit
        if user_input.lower() in ("exit", "quit", "bye", "q"):
            print("Goodbye!")
            break

        # Inject user message into state
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=user_input))
        state["messages"] = messages
        state["last_user_input"] = user_input

        # Run one turn through the graph
        try:
            state = turn_graph.invoke(state, config={"recursion_limit": 50})
        except Exception as e:
            print(f"\nAgent error: {e}")
            print("You can try again with a different question.\n")
            continue

        # Display the response
        _display_response(state)

        # If the graph classified this turn as EXIT, break the REPL.
        if state.get("current_intent") == "EXIT":
            break


def _display_response(state: dict) -> None:
    """Print the most recent AI message to the user."""
    trace_logs = state.get("_turn_debug_logs", [])
    subagent_count = state.get("_turn_subagent_count", 0)

    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            # Handle Responses API list-of-blocks format
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(p for p in text_parts if p)
            if content and isinstance(content, str) and content.strip():
                print(f"\nAssistant> {content.strip()}\n")
                print(f"Trace> subagents_used={subagent_count}")
                for line in trace_logs[:10]:
                    print(f"Trace> {line}")
                print()
                return
    print("\nAssistant> (no response generated)\n")


if __name__ == "__main__":
    main()
