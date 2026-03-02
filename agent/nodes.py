from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.config import AgentConfig
from agent.prompts import REASONER_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT
from agent.state import AgentState
from agent.token_utils import estimate_token_count, trim_text_to_token_budget


def _repo_name_from_url(url: str) -> str:
    """Extract the repository name from a git clone URL."""
    path = urlparse(url).path
    name = Path(path).stem
    return name or "repo"


def initialize_workspace(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    repo_parent = config.repo_dir
    repo_parent.mkdir(parents=True, exist_ok=True)
    repo_name = _repo_name_from_url(config.clone_url)
    repo_path = repo_parent / repo_name

    if repo_path.exists() and (repo_path / ".git").exists():
        summary = f"Workspace already initialized; repository folder exists at {repo_path}."
    else:
        clone_cmd = f"git clone {config.clone_url}"
        result = subprocess.run(clone_cmd, text=True, capture_output=True, shell=True, cwd=str(repo_parent))
        if result.returncode != 0 and "already exists" not in (result.stderr or ""):
            summary = (
                "Workspace initialization attempted but clone failed. "
                f"stderr: {(result.stderr or '').strip()[:300]}"
            )
        else:
            summary = f"Workspace initialized and repository cloned into {repo_path}."

    if repo_path.exists() and repo_path.is_dir():
        os.chdir(repo_path)
        summary = _merge_summary(summary, f"Working directory set to {repo_path}.")

    # --- Environment Detection Phase ---
    env_facts = _probe_environment(repo_path)
    summary = _merge_summary(summary, "Environment: " + " | ".join(env_facts))

    message = HumanMessage(
        content=(
            "Start exploring and building the cloned repository. "
            "First discover the project structure and build system, then build and run tests. "
            "Read the environment facts in the knowledge summary and adapt your build commands accordingly. "
            "Use tools iteratively and stop with a clear final report."
        )
    )

    return {
        "messages": [message],
        "summary_of_knowledge": _merge_summary(state.get("summary_of_knowledge", ""), summary),
        "status": "EXPLORING",
    }


def _probe_environment(repo_path: Path) -> list[str]:
    """Detect OS, available build tools, recommended cmake generator, and fix common issues."""
    env_facts: list[str] = []

    # 1. OS detection
    env_facts.append(f"os={platform.system()}")
    env_facts.append(f"os_name={os.name}")

    # 2. Clean stale build directory to avoid cmake cache conflicts
    build_dir = repo_path / "build"
    if build_dir.exists():
        try:
            shutil.rmtree(build_dir, ignore_errors=True)
            env_facts.append("stale_build_dir=cleaned")
        except Exception:
            env_facts.append("stale_build_dir=cleanup_failed")

    # 3. Windows long paths — enable automatically
    if os.name == "nt":
        try:
            subprocess.run(
                "git config core.longpaths true",
                shell=True, capture_output=True, text=True, timeout=5,
                cwd=str(repo_path),
            )
            env_facts.append("git_longpaths=enabled")
        except Exception:
            env_facts.append("git_longpaths=failed_to_enable")

    # 4. Detect available tools
    tool_checks = [
        ("cmake --version", "cmake"),
        ("g++ --version", "gxx"),
        ("cl", "msvc"),
        ("ninja --version", "ninja"),
        ("mingw32-make --version", "mingw32_make"),
        ("make --version", "make"),
    ]
    for cmd, label in tool_checks:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            available = result.returncode == 0
            first_line = (result.stdout or result.stderr or "").strip().split("\n")[0][:100]
            env_facts.append(f"{label}={'yes: ' + first_line if available else 'not found'}")
        except Exception:
            env_facts.append(f"{label}=check_failed")

    # 5. Detect best cmake generator
    if os.name == "nt":
        if shutil.which("ninja"):
            env_facts.append("recommended_generator=Ninja")
        elif shutil.which("mingw32-make"):
            env_facts.append("recommended_generator=MinGW Makefiles")
        elif shutil.which("cl"):
            env_facts.append("recommended_generator=default (Visual Studio)")
        else:
            env_facts.append("recommended_generator=unknown (no build tool found)")
    else:
        if shutil.which("ninja"):
            env_facts.append("recommended_generator=Ninja")
        else:
            env_facts.append("recommended_generator=default (Unix Makefiles)")

    # 6. Path length risk assessment (Windows 260 char limit)
    path_len = len(str(repo_path.resolve()))
    if path_len > 120:
        env_facts.append(f"path_length_risk=high ({path_len} chars)")
    else:
        env_facts.append(f"path_length_risk=low ({path_len} chars)")

    return env_facts


def agent_reasoner(state: AgentState, config: AgentConfig) -> dict[str, Any]:
    from agent.tools import ALL_TOOLS

    reasoner_output_budget = _reasoner_output_budget(config)

    model = _build_chat_model(config.model_name, reasoner_output_budget).bind_tools(ALL_TOOLS)

    system = SystemMessage(content=REASONER_SYSTEM_PROMPT)
    summary_context = HumanMessage(content=f"Knowledge summary: {state['summary_of_knowledge']}")
    history = list(state["messages"])
    stagnation_hint = _build_stagnation_hint(state, history)
    messages_for_model = _fit_reasoner_messages_to_budget(
        system=system,
        summary_context=summary_context,
        history=history,
        model_name=config.model_name,
        input_budget=config.input_token_budget,
        stagnation_hint=stagnation_hint,
    )
    messages_for_model = _sanitize_tool_message_sequence(messages_for_model)

    response = model.invoke(messages_for_model)

    # Reflective retry: if the model stops on FAILED status, ask it to reconsider with evidence.
    if _should_reflect_on_failure(state, config, response):
        reflection_hint = HumanMessage(content=_build_failure_reflection_hint(state))
        reflected_messages = [*messages_for_model, response, reflection_hint]
        reflected_messages = _fit_messages_to_budget(
            reflected_messages,
            model_name=config.model_name,
            input_budget=config.input_token_budget,
        )
        reflected_messages = _sanitize_tool_message_sequence(reflected_messages)
        reflected_response = model.invoke(reflected_messages)
        if getattr(reflected_response, "tool_calls", None):
            response = reflected_response

    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }


def _should_reflect_on_failure(state: AgentState, config: AgentConfig, response: AIMessage) -> bool:
    """If the model tries to stop after failure, trigger one reflective re-reasoning pass."""
    if getattr(response, "tool_calls", None):
        return False

    if state["status"] not in ("FAILED",):
        return False

    if state["step_count"] >= config.max_steps - 2:
        return False

    if state.get("consecutive_errors", 0) >= config.failure_retry_limit:
        return False

    return True


def _build_failure_reflection_hint(state: AgentState) -> str:
    last_tool = _last_tool_message(state.get("messages", []))
    last_cmd = _extract_cmd(str(last_tool.content)) if last_tool is not None else None
    last_error = _extract_error_signature(last_tool.content) if last_tool is not None else None

    parts = [
        "You attempted to stop while status is FAILED.",
        "Re-evaluate the latest evidence and propose a different next action.",
        "Do not repeat identical discovery calls unless they add new evidence.",
    ]
    if last_cmd:
        parts.append(f"Last command: {last_cmd}")
    if last_error:
        parts.append(f"Last error signal: {last_error}")
    parts.append(
        "Choose the next tool call that most directly reduces uncertainty or addresses the failure root cause."
    )
    return " ".join(parts)


def _last_tool_message(messages: list[BaseMessage]) -> ToolMessage | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message
    return None


def _fit_messages_to_budget(
    messages: list[BaseMessage],
    model_name: str,
    input_budget: int,
) -> list[BaseMessage]:
    current_messages = list(messages)
    while estimate_token_count(current_messages, model_name) > input_budget and len(current_messages) > 2:
        if not _pop_oldest_tool_observation_pair(current_messages):
            current_messages.pop(0)
    return current_messages


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
    command_evidence = _build_command_evidence_snapshot(state)
    report_request = HumanMessage(
        content=(
            f"Status: {state['status']}\n"
            f"Consecutive errors: {state['consecutive_errors']}\n"
            f"Step count: {state['step_count']}\n"
            f"Knowledge summary: {state['summary_of_knowledge']}\n"
            f"Environment facts:\n{environment_facts}\n"
            f"Command evidence:\n{command_evidence}\n"
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

            if "test" in lowered:
                test_summary = _extract_test_summary(text)
                if test_summary:
                    findings.append(test_summary)
            build_summary = _extract_build_summary(text)
            if build_summary:
                findings.append(build_summary)

        tool_error = _extract_error_signature(text)
        if tool_error:
            errors.append(tool_error[:160])

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

    # Treat structured command output as authoritative for errors.
    if "[cmd]=" in content and "[exit_code]=" in content:
        cmd = (_extract_cmd(content) or "").strip()
        code = _extract_exit_code(content)
        if code is not None and code != 0:
            line = _first_matching_line(content, r".*\b(error|failed|fatal)\b.*")
            if line:
                return f"{cmd}: {line}"[:200]
            return f"{cmd}: non-zero exit code {code}"[:200]
        return None

    # For non-shell tool outputs, only count explicit tool validation failures.
    validation_error = _first_matching_line(
        content,
        r"line range too large|invalid line range|invalid regex|invalid file|invalid directory|depth too large",
    )
    if validation_error:
        return validation_error[:200]

    # Ignore incidental words like 'error' found in search results/changelog lines.
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

    # CTest / test runner detection (most specific first)
    if re.search(r"\bctest\b", cmd):
        if exit_code == 0:
            return "SUCCESS"
        if exit_code is not None:
            return "FAILED"
        return "TESTING"

    # Generic test runner in command
    if re.search(r"\btest\b", cmd) and not re.search(r"-D\w*test", cmd, re.IGNORECASE):
        if exit_code == 0:
            return "SUCCESS"
        if exit_code is not None:
            return "FAILED"
        return "TESTING"

    # cmake configure command (cmake -B build ... without --build)
    if re.search(r"\bcmake\b", cmd) and not re.search(r"--build", cmd):
        if exit_code == 0:
            return "CONFIGURING"
        if exit_code is not None:
            return "FAILED"
        return "CONFIGURING"

    # Build command detection
    if re.search(r"\bcmake\s+--build\b|\bmake\b|\bninja\b|\bmsbuild\b|\bmingw32-make\b", cmd):
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
    stagnation_hint: HumanMessage | None = None,
) -> list[BaseMessage]:
    summary_text = str(summary_context.content)
    current_summary = summary_text
    current_history = list(history)

    while True:
        candidate_summary = HumanMessage(content=current_summary)
        candidate_messages = [system, candidate_summary]
        if stagnation_hint is not None:
            candidate_messages.append(stagnation_hint)
        candidate_messages.extend(current_history)
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
        final_messages: list[BaseMessage] = [system, HumanMessage(content=current_summary)]
        if stagnation_hint is not None:
            final_messages.append(stagnation_hint)
        if estimate_token_count(final_messages, model_name) <= input_budget:
            return final_messages


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

    # Always protect the most recent tool-call context so the model can react
    important.update(_recent_tool_context_indexes(messages, keep_pairs=2))

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


def _recent_tool_context_indexes(messages: list[BaseMessage], keep_pairs: int = 2) -> set[int]:
    protected: set[int] = set()
    remaining = keep_pairs

    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if not isinstance(msg, AIMessage) or not getattr(msg, "tool_calls", None):
            continue

        protected.add(index)
        for j in range(index + 1, len(messages)):
            if isinstance(messages[j], ToolMessage):
                protected.add(j)
            elif isinstance(messages[j], AIMessage):
                break

        remaining -= 1
        if remaining <= 0:
            break

    return protected


def _build_stagnation_hint(state: AgentState, history: list[BaseMessage]) -> HumanMessage | None:
    signatures = _recent_tool_call_signatures(history, window=4)
    if len(signatures) < 3:
        signatures = signatures

    # If the same tool-call signature repeats across recent turns, nudge strategy change.
    if len(set(signatures[-3:])) == 1:
        repeated = signatures[-1]
        return HumanMessage(
            content=(
                "Stagnation detected: the same tool-call pattern was repeated for 3 consecutive turns "
                f"({repeated}). Choose a different strategy using new evidence. "
                "Do not repeat identical discovery calls unless a command failure explicitly requires it. "
                "Prefer the next unresolved lifecycle step (configure/build/test or failure diagnosis)."
            )
        )

    recent_patterns = _recent_tool_name_patterns(history, window=5)
    if recent_patterns and all("execute_shell_command" not in pattern for pattern in recent_patterns):
        return HumanMessage(
            content=(
                "Stagnation detected: multiple turns used discovery/search tools only and no real command execution. "
                "Pick a concrete next experiment using execute_shell_command based on current evidence, "
                "then use its output to update strategy."
            )
        )

    # Progress-aware guard: if several shell commands ran but none advanced lifecycle, force pivot.
    recent_shell_cmds = _recent_shell_commands(history, window=8)
    if state.get("step_count", 0) >= 3 and recent_shell_cmds:
        has_lifecycle_cmd = any(_is_lifecycle_command(cmd) for cmd in recent_shell_cmds)
        if not has_lifecycle_cmd:
            cmds = " ; ".join(recent_shell_cmds[-4:])
            return HumanMessage(
                content=(
                    "Stagnation detected: recent shell commands did not perform configure/build/test actions. "
                    f"Recent shell commands: {cmds}. "
                    "Choose one command that advances lifecycle (configure, build, or test) based on detected build files and tools. "
                    "Avoid repository metadata commands unless they directly help diagnose a failure."
                )
            )

    # If tool usage stays discovery-heavy for too long, request a lifecycle action.
    if state.get("step_count", 0) >= 6 and recent_patterns:
        discovery_only = True
        for pattern in recent_patterns[-4:]:
            names = [n for n in pattern.split("|") if n]
            if any(name not in {"list_directory", "search_codebase", "read_file_chunk"} for name in names):
                discovery_only = False
                break
        if discovery_only:
            return HumanMessage(
                content=(
                    "Stagnation detected: last turns were discovery-only. "
                    "Use current evidence to run a concrete build-system command and evaluate its output."
                )
            )

    return None


def _recent_tool_call_signatures(history: list[BaseMessage], window: int = 4) -> list[str]:
    signatures: list[str] = []
    for message in history:
        if not isinstance(message, AIMessage):
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            continue

        parts: list[str] = []
        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args", {})
            args_repr = str(args)
            parts.append(f"{name}:{args_repr}")
        signatures.append(" | ".join(parts))

    if window <= 0:
        return signatures
    return signatures[-window:]


def _recent_tool_name_patterns(history: list[BaseMessage], window: int = 5) -> list[str]:
    patterns: list[str] = []
    for message in history:
        if not isinstance(message, AIMessage):
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            continue
        names = sorted(str(call.get("name", "")) for call in tool_calls)
        patterns.append("|".join(names))
    if window <= 0:
        return patterns
    return patterns[-window:]


def _recent_shell_commands(history: list[BaseMessage], window: int = 8) -> list[str]:
    cmds: list[str] = []
    for message in history:
        if not isinstance(message, ToolMessage):
            continue
        text = str(message.content)
        cmd = _extract_cmd(text)
        if cmd:
            cmds.append(cmd)
    if window <= 0:
        return cmds
    return cmds[-window:]


def _is_lifecycle_command(cmd: str) -> bool:
    lowered = cmd.lower()
    patterns = [
        r"\bcmake\b.*\b(-b|-s|--build)\b",
        r"\bctest\b",
        r"\bninja\b",
        r"\bmingw32-make\b",
        r"\bmake\b",
        r"\bmeson\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_important_tool_output(text: str) -> bool:
    cmd = (_extract_cmd(text) or "").lower()
    lowered = text.lower()

    # Protect any test runner output
    if re.search(r"\btest\b", cmd):
        return True

    # Protect messages containing test results or failure signals
    signal_patterns = [
        r"\d+% tests passed",
        r"\d+ (?:tests?\s+)?passed",
        r"\d+ (?:tests?\s+)?failed",
        r"the following tests? failed",
        r"\*\*\*(timeout|failed)",
        r"assertion.*(failed|error)",
        r"\[exit_code\]=[1-9]\d*",
    ]
    for pattern in signal_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True

    # Protect successful build outputs
    if re.search(r"\b(build|compile|make|ninja)\b", cmd) and "[exit_code]=0" in lowered:
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
    # CTest-style: "N% tests passed, M tests failed out of K"
    total_line = _first_matching_line(text, r"\d+% tests passed, \d+ tests failed out of \d+")
    if total_line:
        return total_line
    # Generic pass/fail summaries
    pass_line = _first_matching_line(text, r"\d+\s+(?:tests?\s+)?passed")
    if pass_line:
        return pass_line
    fail_line = _first_matching_line(text, r"\d+\s+(?:tests?\s+)?failed")
    if fail_line:
        return fail_line
    return None


def _extract_build_summary(text: str) -> str | None:
    markers = len(re.findall(r"Built target |\[\d+/\d+\]", text))
    if markers > 0:
        return f"Build steps observed: {markers}"
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


def _collect_environment_facts(config: AgentConfig) -> str:
    facts: list[str] = []
    cwd = Path.cwd()
    repo_name = _repo_name_from_url(config.clone_url)
    repo_path = config.repo_dir / repo_name

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


def _build_command_evidence_snapshot(state: AgentState) -> str:
    """Scan retained tool messages for command outputs and build a structured evidence summary."""
    tool_messages = [msg for msg in state.get("messages", []) if isinstance(msg, ToolMessage)]
    records: list[str] = []

    for message in tool_messages:
        text = str(message.content)
        cmd = _extract_cmd(text)
        if not cmd:
            continue

        exit_code = _extract_exit_code(text)
        test_summary = _extract_test_summary(text)
        error_line = _first_matching_line(text, r".*\b(error|failed|fatal)\b.*")

        parts = [f"cmd={cmd}"]
        if exit_code is not None:
            parts.append(f"exit_code={exit_code}")
        if test_summary:
            parts.append(f"test_result={test_summary}")
        if error_line and (exit_code is None or exit_code != 0):
            parts.append(f"error={error_line[:160]}")
        records.append(" | ".join(parts))

    if not records:
        return "No command output found in retained messages."
    return "\n".join(records)
