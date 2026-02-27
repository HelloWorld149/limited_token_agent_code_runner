from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import uuid4

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

    reasoner_output_budget = _reasoner_output_budget(config)

    model = _build_chat_model(config.model_name, reasoner_output_budget).bind_tools(ALL_TOOLS)

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

    if _should_force_failure_retry(state, response, config):
        response = _build_forced_retry_tool_call(state)

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
    summary = _cap_summary(summary, config)

    consecutive_errors = _compute_consecutive_errors(state)
    status = _infer_status(state)

    return {
        "messages": messages,
        "summary_of_knowledge": summary,
        "consecutive_errors": consecutive_errors,
        "status": status,
    }


def generate_report(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    report_output_budget = _report_output_budget(config)

    model = _build_chat_model(config.model_name, report_output_budget)
    system = SystemMessage(content=REPORT_SYSTEM_PROMPT)
    environment_facts = _collect_environment_facts(config)
    report_request = HumanMessage(
        content=(
            f"Status: {state['status']}\n"
            f"Consecutive errors: {state['consecutive_errors']}\n"
            f"Step count: {state['step_count']}\n"
            f"Knowledge summary: {state['summary_of_knowledge']}\n"
            f"Environment facts:\n{environment_facts}\n"
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
    if state["step_count"] >= config.max_steps:
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
    if not messages:
        return "Pruned context contained no actionable tool details."

    findings: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    files: list[str] = []

    for message in messages:
        text = str(message.content)
        lowered = text.lower()

        cmd_line = _extract_cmd(text)
        if cmd_line:
            commands.append(cmd_line[:160])

        if "[exit_code]" in lowered:
            exit_line = _first_matching_line(text, r"\[exit_code\]\s*=\s*\d+|\[exit_code\]=\d+")
            if exit_line:
                findings.append(exit_line)

            if "ctest" in lowered:
                test_summary = _extract_test_summary(text)
                if test_summary:
                    findings.append(test_summary)
            if "cmake --build" in lowered or "ninja" in lowered or "make" in lowered:
                build_summary = _extract_build_summary(text)
                if build_summary:
                    findings.append(build_summary)

        error_line = _first_matching_line(text, r".*\b(error|failed|fatal)\b.*")
        if error_line:
            errors.append(error_line[:160])

        file_line = _first_matching_line(text, r"[\w\-./\\]+\.(cpp|cc|cxx|h|hpp|cmake|txt):\d+")
        if file_line:
            files.append(file_line[:140])

    parts: list[str] = []
    if commands:
        parts.append("commands=" + " ; ".join(_unique_keep_order(commands)[:2]))
    if findings:
        parts.append("results=" + " ; ".join(_unique_keep_order(findings)[:2]))
    if errors:
        parts.append("errors=" + " ; ".join(_unique_keep_order(errors)[:2]))
    if files:
        parts.append("files=" + " ; ".join(_unique_keep_order(files)[:2]))

    if not parts:
        return "Pruned tool context had no retained build/test/error signals."
    return "Pruned context summary: " + " | ".join(parts) + "."


def _merge_summary(existing: str, update: str) -> str:
    if not existing:
        return update
    return f"{existing} {update}".strip()


def _cap_summary(summary: str, config: AgentConfig) -> str:
    max_summary_tokens = max(256, min(1200, config.input_token_budget // 3))
    return trim_text_to_token_budget(summary, config.model_name, max_summary_tokens)


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

    last_text = str(tool_messages[-1].content)
    last = last_text.lower()
    if "[cmd]=" not in last and "[exit_code]=" not in last:
        return state["status"]

    cmd = (_extract_cmd(last_text) or "").lower()
    exit_code = _extract_exit_code(last_text)

    if "ctest" in cmd:
        if exit_code == 0:
            return "SUCCESS"
        return "FAILED"

    if "cmake --build" in cmd or "ninja" in cmd or re.search(r"\bmake\b", cmd):
        if exit_code is None:
            return "BUILDING"
        return "FAILED" if exit_code != 0 else "BUILDING"

    if exit_code is not None:
        return "FAILED" if exit_code != 0 else state["status"]

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


def _reasoner_output_budget(config: AgentConfig) -> int:
    return max(256, min(700, config.output_token_budget // 2))


def _report_output_budget(config: AgentConfig) -> int:
    return max(512, config.output_token_budget)


def _first_matching_line(text: str, pattern: str) -> str | None:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for line in text.splitlines():
        if regex.search(line):
            return line.strip()
    return None


def _extract_cmd(text: str) -> str | None:
    line = _first_matching_line(text, r"\[cmd\]=.+")
    if not line:
        return None
    return line.split("=", 1)[1].strip()


def _extract_exit_code(text: str) -> int | None:
    line = _first_matching_line(text, r"\[exit_code\]\s*=\s*\d+|\[exit_code\]=\d+")
    if not line:
        return None
    match = re.search(r"(\d+)", line)
    if not match:
        return None
    return int(match.group(1))


def _extract_test_summary(text: str) -> str | None:
    total_line = _first_matching_line(text, r"\d+% tests passed, \d+ tests failed out of \d+")
    if total_line:
        return total_line
    pass_line = _first_matching_line(text, r"100% tests passed")
    if pass_line:
        return pass_line
    return None


def _extract_build_summary(text: str) -> str | None:
    built_targets = len(re.findall(r"Built target ", text))
    if built_targets > 0:
        return f"Built targets observed: {built_targets}"
    return None


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _build_chat_model(model_name: str, max_tokens: int) -> ChatOpenAI:
    kwargs = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    if _is_codex_or_gpt5_model(model_name):
        try:
            return ChatOpenAI(**kwargs, use_responses_api=True)
        except TypeError as exc:
            raise RuntimeError(
                "Codex/GPT-5 models require Responses API support in your langchain-openai version. "
                "Please upgrade langchain-openai and openai packages, then retry."
            ) from exc

    return ChatOpenAI(**kwargs)


def _is_codex_or_gpt5_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "codex" in lowered or lowered.startswith("gpt-5")


def _should_force_failure_retry(state: AgentState, response: AIMessage, config: AgentConfig) -> bool:
    has_tool_call = bool(getattr(response, "tool_calls", None))
    if has_tool_call:
        return False
    if state.get("status") != "FAILED":
        return False
    if state.get("step_count", 0) >= config.max_steps - 1:
        return False
    if state.get("consecutive_errors", 0) >= config.failure_retry_limit:
        return False
    return True


def _build_forced_retry_tool_call(state: AgentState) -> AIMessage:
    if state.get("consecutive_errors", 0) == 0:
        cmd = "ctest --test-dir build --output-on-failure -j1"
    else:
        cmd = "ctest --test-dir build -R \"fetch_content|regression1|testsuites|class_parser\" --output-on-failure -V"

    return AIMessage(
        content="Auto-retry policy: failure detected, running one more diagnostic command before finalizing.",
        tool_calls=[
            {
                "name": "execute_shell_command",
                "args": {"cmd": cmd},
                "id": f"forced_retry_{uuid4().hex[:10]}",
                "type": "tool_call",
            }
        ],
    )


def _collect_environment_facts(config: AgentConfig) -> str:
    facts: list[str] = []
    cwd = Path.cwd()
    repo_path = config.repo_dir / "json"

    facts.append(f"cwd={cwd}")
    facts.append(f"cwd_path_length={len(str(cwd))}")
    facts.append(f"repo_path={repo_path}")
    facts.append(f"repo_path_length={len(str(repo_path))}")
    facts.append(f"os_name={os.name}")

    for cmd, label in [
        ("python --version", "python"),
        ("cmake --version", "cmake"),
        ("g++ --version", "gxx"),
        ("git --version", "git"),
        ("git config --get core.longpaths", "git_core_longpaths"),
    ]:
        output = _run_quick_command(cmd)
        if output:
            facts.append(f"{label}={output}")

    if len(str(repo_path)) > 120:
        facts.append("path_length_risk=high")
    else:
        facts.append("path_length_risk=low")

    return "\n".join(facts)


def _run_quick_command(cmd: str) -> str:
    try:
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=5)
    except Exception:
        return ""

    text = (result.stdout or result.stderr or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return first_line[:200]
