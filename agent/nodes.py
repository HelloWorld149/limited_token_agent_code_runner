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
    messages_for_model = _sanitize_tool_message_sequence(messages_for_model)

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
        protected_indexes = _important_message_indexes(messages)
        pair = _pop_oldest_tool_observation_pair(messages, protected_indexes)
        if pair:
            dropped.extend(pair)
        else:
            removed = _pop_oldest_non_protected_message(messages, protected_indexes)
            if removed is None and messages:
                removed = messages.pop(0)
            if removed is not None:
                dropped.append(removed)
            else:
                break
        token_count = estimate_token_count(messages, config.model_name)

    messages = _sanitize_tool_message_sequence(messages)

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
    ctest_snapshot = _build_ctest_evidence_snapshot(state)
    report_request = HumanMessage(
        content=(
            f"Status: {state['status']}\n"
            f"Consecutive errors: {state['consecutive_errors']}\n"
            f"Step count: {state['step_count']}\n"
            f"Knowledge summary: {state['summary_of_knowledge']}\n"
            f"Environment facts:\n{environment_facts}\n"
            f"CTest evidence snapshot:\n{ctest_snapshot}\n"
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
    report_messages = _sanitize_tool_message_sequence(report_messages)
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

        protected_indexes = _important_message_indexes(current_history)
        pair = _pop_oldest_tool_observation_pair(current_history, protected_indexes)
        if pair:
            continue
        removed = _pop_oldest_non_protected_message(current_history, protected_indexes)
        if removed is not None:
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

        protected_indexes = _important_message_indexes(current_history)
        pair = _pop_oldest_tool_observation_pair(current_history, protected_indexes)
        if pair:
            continue
        removed = _pop_oldest_non_protected_message(current_history, protected_indexes)
        if removed is not None:
            continue
        if current_history:
            current_history.pop(0)
            continue
        request_text = trim_text_to_token_budget(str(report_request.content), model_name, max(96, input_budget // 2))
        return [system, HumanMessage(content=request_text)]


def _pop_oldest_tool_observation_pair(
    messages: list[BaseMessage],
    protected_indexes: set[int] | None = None,
) -> list[BaseMessage]:
    protected = protected_indexes or set()

    ai_index = None
    pair_indexes: list[int] = []

    for index, message in enumerate(messages):
        if not (isinstance(message, AIMessage) and getattr(message, "tool_calls", None)):
            continue

        candidate_indexes = [index]
        for tool_index in range(index + 1, len(messages)):
            if isinstance(messages[tool_index], ToolMessage):
                candidate_indexes.append(tool_index)
            elif isinstance(messages[tool_index], AIMessage):
                break

        if any(candidate_index in protected for candidate_index in candidate_indexes):
            continue

        ai_index = index
        pair_indexes = candidate_indexes
        break

    if ai_index is None:
        return []

    removed: list[BaseMessage] = []
    for index in sorted(pair_indexes, reverse=True):
        removed.append(messages.pop(index))

    removed.reverse()
    return removed


def _pop_oldest_non_protected_message(
    messages: list[BaseMessage],
    protected_indexes: set[int],
) -> BaseMessage | None:
    for index, _message in enumerate(messages):
        if index in protected_indexes:
            continue
        return messages.pop(index)
    return None


def _important_message_indexes(messages: list[BaseMessage]) -> set[int]:
    important: set[int] = set()

    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue

        text = str(message.content)
        if not _is_important_tool_output(text):
            continue

        important.add(index)

        for prev_index in range(index - 1, -1, -1):
            prev_message = messages[prev_index]
            if isinstance(prev_message, AIMessage) and getattr(prev_message, "tool_calls", None):
                important.add(prev_index)
                break
            if isinstance(prev_message, ToolMessage):
                break

    return important


def _is_important_tool_output(text: str) -> bool:
    cmd = (_extract_cmd(text) or "").lower()
    lowered = text.lower()

    if "ctest" in cmd and "-r" not in cmd:
        return True

    signal_patterns = [
        r"\d+% tests passed, \d+ tests failed out of \d+",
        r"the following tests failed",
        r"\*\*\*timeout",
        r"did not throw",
        r"\bis not correct\b",
        r"\[exit_code\]=[1-9]\d*",
    ]
    for pattern in signal_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True

    if "cmake --build" in cmd and "[exit_code]=0" in lowered:
        return True

    return False


def _sanitize_tool_message_sequence(messages: list[BaseMessage]) -> list[BaseMessage]:
    allowed_call_ids: set[str] = set()
    sanitized: list[BaseMessage] = []

    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in getattr(message, "tool_calls", []) or []:
                call_id = str(tool_call.get("id", "")).strip()
                if call_id:
                    allowed_call_ids.add(call_id)
            sanitized.append(message)
            continue

        if isinstance(message, ToolMessage):
            call_id = str(getattr(message, "tool_call_id", "") or "").strip()
            if call_id and call_id in allowed_call_ids:
                sanitized.append(message)
            continue

        sanitized.append(message)

    return sanitized


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


def _build_ctest_evidence_snapshot(state: AgentState) -> str:
    tool_messages = [msg for msg in state.get("messages", []) if isinstance(msg, ToolMessage)]
    ctest_records: list[dict[str, Any]] = []

    for message in tool_messages:
        text = str(message.content)
        cmd = _extract_cmd(text)
        if not cmd or "ctest" not in cmd.lower():
            continue

        summary = _extract_test_summary(text) or "<no test summary line found>"
        failures = _extract_failed_tests(text)
        if not failures:
            failures = _extract_inline_failed_tests(text)

        ctest_records.append(
            {
                "cmd": cmd,
                "summary": summary,
                "failures": failures,
                "is_targeted": bool(re.search(r"(^|\s)-R(\s|=)", cmd)),
            }
        )

    if not ctest_records:
        return "No ctest command output found in retained messages."

    full_runs = [record for record in ctest_records if not record["is_targeted"]]
    targeted_runs = [record for record in ctest_records if record["is_targeted"]]

    latest_full = full_runs[-1] if full_runs else None
    latest_targeted = targeted_runs[-1] if targeted_runs else None

    lines: list[str] = []
    lines.append(f"ctest_runs_observed={len(ctest_records)}")

    if latest_full:
        lines.append(f"full_suite_cmd={latest_full['cmd']}")
        lines.append(f"full_suite_summary={latest_full['summary']}")
        lines.extend(_format_failure_type_lines("full_suite", latest_full["failures"]))
    else:
        lines.append("full_suite_summary=<missing in retained context>")

    if latest_targeted:
        lines.append(f"targeted_cmd={latest_targeted['cmd']}")
        lines.append(f"targeted_summary={latest_targeted['summary']}")
        lines.extend(_format_failure_type_lines("targeted", latest_targeted["failures"]))
    else:
        lines.append("targeted_summary=<none observed>")

    if latest_full and latest_targeted:
        full_names = {entry["name"] for entry in latest_full["failures"]}
        targeted_names = {entry["name"] for entry in latest_targeted["failures"]}
        missing_from_targeted = sorted(name for name in full_names if name and name not in targeted_names)
        extra_in_targeted = sorted(name for name in targeted_names if name and name not in full_names)

        lines.append(f"consistency_missing_from_targeted={missing_from_targeted or ['<none>']}")
        lines.append(f"consistency_extra_in_targeted={extra_in_targeted or ['<none>']}")

    lines.append("report_rule=prioritize full_suite_* lines over targeted_* lines when totals differ")
    return "\n".join(lines)


def _extract_failed_tests(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    section_index = None
    for index, line in enumerate(lines):
        if "The following tests FAILED" in line:
            section_index = index
            break

    if section_index is None:
        return []

    results: list[dict[str, str]] = []
    pattern = re.compile(r"^\s*\d+\s*-\s*([^\(]+?)\s*\((Failed|Timeout)\)")
    for line in lines[section_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        match = pattern.search(stripped)
        if not match:
            if stripped.lower().startswith("errors while running ctest"):
                break
            continue
        results.append({"name": match.group(1).strip(), "type": match.group(2).strip()})
    return results


def _extract_inline_failed_tests(text: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    pattern = re.compile(r"Test\s+#?\d+:\s*([\w\-\.]+)\s*\.*\*\*\*(Failed|Timeout)", re.IGNORECASE)

    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(1).strip()
        failure_type = match.group(2).capitalize()
        key = (name, failure_type)
        if key in seen:
            continue
        seen.add(key)
        results.append({"name": name, "type": failure_type})

    return results


def _format_failure_type_lines(prefix: str, failures: list[dict[str, str]]) -> list[str]:
    if not failures:
        return [f"{prefix}_failed_tests=[]", f"{prefix}_failure_type_counts={{}}"]

    names = sorted({entry["name"] for entry in failures if entry.get("name")})
    counts: dict[str, int] = {}
    for entry in failures:
        failure_type = entry.get("type", "Unknown")
        counts[failure_type] = counts.get(failure_type, 0) + 1

    return [
        f"{prefix}_failed_tests={names}",
        f"{prefix}_failure_type_counts={counts}",
    ]
