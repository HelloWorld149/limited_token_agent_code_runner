from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.config import AgentConfig
from agent.prompts import REASONER_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT
from agent.state import AgentState
from agent.token_utils import estimate_token_count, trim_text_to_token_budget


def initialize_workspace(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    repo_parent = config.repo_dir
    repo_parent.mkdir(parents=True, exist_ok=True)
    repo_path = repo_parent / "json"

    if repo_path.exists() and (repo_path / ".git").exists():
        summary = "Workspace already initialized; repository folder exists at workspace/json."
    else:
        clone_cmd = "git clone https://github.com/nlohmann/json"
        result = subprocess.run(clone_cmd, text=True, capture_output=True, shell=True, cwd=str(repo_parent))
        if result.returncode != 0 and "already exists" not in (result.stderr or ""):
            summary = (
                "Workspace initialization attempted but clone failed. "
                f"stderr: {(result.stderr or '').strip()[:300]}"
            )
        else:
            summary = "Workspace initialized and repository cloned into workspace/json."

    if repo_path.exists() and repo_path.is_dir():
        os.chdir(repo_path)
        summary = _merge_summary(summary, f"Working directory set to {repo_path}.")

    message = HumanMessage(
        content=(
            "Start exploring and building the cloned nlohmann/json repository. "
            "Use tools iteratively and stop with a clear final report."
        )
    )

    return {
        "messages": [message],
        "summary_of_knowledge": _merge_summary(state.get("summary_of_knowledge", ""), summary),
        "status": "EXPLORING",
    }


def agent_reasoner(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    from agent.tools import ALL_TOOLS

    model = ChatOpenAI(
        model=config.model_name,
        temperature=0,
        max_tokens=config.output_token_budget,
    ).bind_tools(ALL_TOOLS)

    system = SystemMessage(content=REASONER_SYSTEM_PROMPT)
    summary_context = HumanMessage(content=f"Knowledge summary: {state['summary_of_knowledge']}")
    history = list(state["messages"])
    messages_for_model = _fit_reasoner_messages_to_budget(
        system=system,
        summary_context=summary_context,
        history=history,
        model_name=config.model_name,
        input_budget=config.input_token_budget,
    )

    response = model.invoke(messages_for_model)
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def context_manager(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    messages = list(state["messages"])
    token_count = estimate_token_count(messages, config.model_name)
    dropped: list[BaseMessage] = []

    while token_count > config.prune_threshold and len(messages) > 4:
        pair = _pop_oldest_tool_observation_pair(messages)
        if pair:
            dropped.extend(pair)
        else:
            dropped.append(messages.pop(0))
        token_count = estimate_token_count(messages, config.model_name)

    summary = state["summary_of_knowledge"]
    if dropped:
        summary_update = _summarize_messages(dropped)
        summary = _merge_summary(summary, summary_update)

    consecutive_errors = _compute_consecutive_errors(state)
    status = _infer_status(state)

    return {
        "messages": messages,
        "summary_of_knowledge": summary,
        "consecutive_errors": consecutive_errors,
        "status": status,
    }


def generate_report(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    model = ChatOpenAI(
        model=config.model_name,
        temperature=0,
        max_tokens=config.output_token_budget,
    )
    system = SystemMessage(content=REPORT_SYSTEM_PROMPT)
    report_request = HumanMessage(
        content=(
            f"Status: {state['status']}\n"
            f"Consecutive errors: {state['consecutive_errors']}\n"
            f"Step count: {state['step_count']}\n"
            f"Knowledge summary: {state['summary_of_knowledge']}\n"
            "Use recent messages as evidence and produce the final report."
        )
    )
    report_messages = _fit_report_messages_to_budget(
        system=system,
        history=list(state["messages"]),
        report_request=report_request,
        model_name=config.model_name,
        input_budget=config.input_token_budget,
    )
    response = model.invoke(report_messages)
    return {"messages": [response]}


def route_from_reasoner(state: AgentState, config: AgentConfig) -> str:
    if state["step_count"] > config.max_steps:
        return "generate_report"

    last = _last_ai_message(state["messages"])
    if last and getattr(last, "tool_calls", None):
        return "execute_tools"
    return "generate_report"


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _summarize_messages(messages: list[BaseMessage]) -> str:
    snippets: list[str] = []
    for message in messages[-4:]:
        text = str(message.content).replace("\n", " ").strip()
        if text:
            snippets.append(text[:120])
    if not snippets:
        return "Older tool interactions were pruned to preserve context budget."
    return "Pruned tool context covered: " + " | ".join(snippets) + "."


def _merge_summary(existing: str, update: str) -> str:
    if not existing:
        return update
    return f"{existing} {update}".strip()


def _compute_consecutive_errors(state: AgentState) -> int:
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    if not tool_messages:
        return state.get("consecutive_errors", 0)

    last_signature = _extract_error_signature(tool_messages[-1].content)
    if not last_signature:
        return 0

    previous = state.get("consecutive_errors", 0)
    if len(tool_messages) >= 2:
        prev_signature = _extract_error_signature(tool_messages[-2].content)
        if prev_signature == last_signature:
            return previous + 1
    return 1


def _extract_error_signature(text: Any) -> str | None:
    content = str(text)
    lines = content.splitlines()
    for line in lines:
        if re.search(r"error|failed|fatal", line, flags=re.IGNORECASE):
            return line.strip()[:200]
    return None


def _infer_status(state: AgentState) -> str:
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    if not tool_messages:
        return state["status"]

    last = str(tool_messages[-1].content).lower()
    if "ctest" in last or "test" in last:
        if "100% tests passed" in last or "passed" in last and "failed" not in last:
            return "SUCCESS"
        return "TESTING"
    if "cmake" in last or "ninja" in last or "build" in last:
        if "error" in last or "failed" in last:
            return "FAILED"
        return "BUILDING"
    return state["status"]


def _fit_reasoner_messages_to_budget(
    system: SystemMessage,
    summary_context: HumanMessage,
    history: list[BaseMessage],
    model_name: str,
    input_budget: int,
) -> list[BaseMessage]:
    summary_text = str(summary_context.content)
    current_summary = summary_text
    current_history = list(history)

    while True:
        candidate_summary = HumanMessage(content=current_summary)
        candidate_messages = [system, candidate_summary, *current_history]
        tokens = estimate_token_count(candidate_messages, model_name)
        if tokens <= input_budget:
            return candidate_messages

        pair = _pop_oldest_tool_observation_pair(current_history)
        if pair:
            continue
        if current_history:
            current_history.pop(0)
            continue

        target_summary_tokens = max(64, input_budget // 2)
        current_summary = trim_text_to_token_budget(current_summary, model_name, target_summary_tokens)
        if estimate_token_count([system, HumanMessage(content=current_summary)], model_name) <= input_budget:
            return [system, HumanMessage(content=current_summary)]


def _fit_report_messages_to_budget(
    system: SystemMessage,
    history: list[BaseMessage],
    report_request: HumanMessage,
    model_name: str,
    input_budget: int,
) -> list[BaseMessage]:
    current_history = list(history)
    while True:
        candidate = [system, *current_history, report_request]
        tokens = estimate_token_count(candidate, model_name)
        if tokens <= input_budget:
            return candidate

        pair = _pop_oldest_tool_observation_pair(current_history)
        if pair:
            continue
        if current_history:
            current_history.pop(0)
            continue
        request_text = trim_text_to_token_budget(str(report_request.content), model_name, max(96, input_budget // 2))
        return [system, HumanMessage(content=request_text)]


def _pop_oldest_tool_observation_pair(messages: list[BaseMessage]) -> list[BaseMessage]:
    ai_index = None
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            ai_index = index
            break

    if ai_index is None:
        return []

    tool_indexes: list[int] = []
    for index in range(ai_index + 1, len(messages)):
        if isinstance(messages[index], ToolMessage):
            tool_indexes.append(index)
        elif isinstance(messages[index], AIMessage):
            break

    removed: list[BaseMessage] = []
    if tool_indexes:
        for index in sorted(tool_indexes, reverse=True):
            removed.append(messages.pop(index))

    removed.append(messages.pop(ai_index))
    removed.reverse()
    return removed
