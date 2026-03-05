from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import AgentConfig
from agent.nodes import (
    answer_question,
    classify_and_prepare,
    continue_or_respond,
    explore_codebase,
    handle_tool_result,
    index_workspace,
    retrieve_context,
    route_after_llm,
    route_by_intent,
    run_build,
    run_tests,
)
from agent.state import AgentState
from agent.tools import ALL_TOOLS


def build_init_graph(config: AgentConfig):
    """Build a one-shot graph that indexes the workspace at startup."""
    graph = StateGraph(AgentState)
    graph.add_node(
        "index_workspace", lambda state: index_workspace(state, config)
    )
    graph.add_edge(START, "index_workspace")
    graph.add_edge("index_workspace", END)
    return graph.compile()


def build_turn_graph(config: AgentConfig):
    """Build a per-turn graph that processes one user message.

    Flow per turn:
        START -> classify_and_prepare -> retrieve_context -> route_by_intent:
            ├── answer_question -> route_after_llm:
            │       ├── execute_tools -> handle_tool_result -> continue_or_respond -> route_after_llm: ...
            │       └── END
            ├── run_build -> route_after_llm: (same)
            ├── run_tests -> route_after_llm: (same)
            ├── explore_codebase -> route_after_llm: (same)
            └── exit -> END
    """
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node(
        "classify_and_prepare",
        lambda state: classify_and_prepare(state, config),
    )
    graph.add_node(
        "retrieve_context", lambda state: retrieve_context(state, config)
    )
    graph.add_node(
        "answer_question", lambda state: answer_question(state, config)
    )
    graph.add_node(
        "run_build", lambda state: run_build(state, config)
    )
    graph.add_node(
        "run_tests", lambda state: run_tests(state, config)
    )
    graph.add_node(
        "explore_codebase", lambda state: explore_codebase(state, config)
    )
    graph.add_node("execute_tools", ToolNode(ALL_TOOLS))
    graph.add_node(
        "handle_tool_result", lambda state: handle_tool_result(state, config)
    )
    graph.add_node(
        "continue_or_respond",
        lambda state: continue_or_respond(state, config),
    )

    # --- Edges ---
    graph.add_edge(START, "classify_and_prepare")
    graph.add_edge("classify_and_prepare", "retrieve_context")

    # Route by intent
    graph.add_conditional_edges(
        "retrieve_context",
        lambda state: route_by_intent(state),
        {
            "answer_question": "answer_question",
            "run_build": "run_build",
            "run_tests": "run_tests",
            "explore_codebase": "explore_codebase",
            "exit": END,
        },
    )

    # ALL intents (including QUESTION) use a ReAct tool loop.
    # The LLM simply chooses not to call tools when pre-retrieved context suffices.
    for node_name in ("answer_question", "run_build", "run_tests", "explore_codebase", "continue_or_respond"):
        graph.add_conditional_edges(
            node_name,
            lambda state: route_after_llm(state),
            {
                "execute_tools": "execute_tools",
                "respond_to_user": END,
            },
        )

    # Tool execution loop
    graph.add_edge("execute_tools", "handle_tool_result")
    graph.add_edge("handle_tool_result", "continue_or_respond")

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graphs for `langgraph dev` / LangGraph Studio.
# langgraph.json points to these variables.
# ---------------------------------------------------------------------------
_default_config = AgentConfig()

turn_graph = build_turn_graph(_default_config)
init_graph = build_init_graph(_default_config)
