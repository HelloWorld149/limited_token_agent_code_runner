from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import AgentConfig
from agent.nodes import (
    agent_reasoner,
    context_manager,
    generate_report,
    initialize_workspace,
    route_from_reasoner,
)
from agent.state import AgentState
from agent.tools import ALL_TOOLS


def build_graph(config: AgentConfig):
    graph = StateGraph(AgentState)

    graph.add_node("initialize_workspace", lambda state: initialize_workspace(state, config))
    graph.add_node("agent_reasoner", lambda state: agent_reasoner(state, config))
    graph.add_node("execute_tools", ToolNode(ALL_TOOLS))
    graph.add_node("context_manager", lambda state: context_manager(state, config))
    graph.add_node("generate_report", lambda state: generate_report(state, config))

    graph.add_edge(START, "initialize_workspace")
    graph.add_edge("initialize_workspace", "agent_reasoner")
    graph.add_conditional_edges(
        "agent_reasoner",
        lambda state: route_from_reasoner(state, config),
        {
            "execute_tools": "execute_tools",
            "generate_report": "generate_report",
        },
    )
    graph.add_edge("execute_tools", "context_manager")
    graph.add_edge("context_manager", "agent_reasoner")
    graph.add_edge("generate_report", END)

    return graph.compile()
