from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re
from typing import Callable
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import AgentConfig
from agent.graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-constrained build agent")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum reasoning loops")
    parser.add_argument("--repo-dir", default=None, help="Local workspace directory")
    parser.add_argument("--clone-url", default=None, help="Git repository URL to clone")
    parser.add_argument("--verbose-loop", action="store_true", help="Print per-loop node updates")
    parser.add_argument("--log-file", default="", help="Optional path to write runtime logs")
    return parser.parse_args()


def _truncate_text(value: str, max_chars: int = 1000) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 24] + "\n... <truncated for display>"


def _format_message_excerpt(message_obj: object) -> str:
    content = str(getattr(message_obj, "content", ""))
    return _truncate_text(content.strip() or "<empty>", max_chars=900)


def _extract_cmd(text: str) -> str | None:
    match = re.search(r"^\[cmd\]=(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_exit_code(text: str) -> int | None:
    match = re.search(r"\[exit_code\]\s*=\s*(\d+)|\[exit_code\]=(\d+)", text)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value)


def _extract_first_error_line(text: str) -> str | None:
    for line in text.splitlines():
        if re.search(r"\berror\b|\bfatal\b|\bfailed\b", line, flags=re.IGNORECASE):
            return line.strip()
    return None


def _extract_test_summary(text: str) -> str | None:
    match = re.search(r"\d+% tests passed, \d+ tests failed out of \d+", text, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    if re.search(r"100% tests passed", text, flags=re.IGNORECASE):
        return "100% tests passed"
    return None


def _extract_build_summary(text: str) -> str | None:
    count = len(re.findall(r"Built target ", text))
    if count > 0:
        return f"Built targets observed: {count}"
    return None


def _print_tool_result_summary(text: str, emit: Callable[[str], None]) -> None:
    cmd = _extract_cmd(text)
    exit_code = _extract_exit_code(text)
    test_summary = _extract_test_summary(text)
    build_summary = _extract_build_summary(text)
    error_line = _extract_first_error_line(text)

    if cmd:
        emit(f"command={cmd}")
    if exit_code is not None:
        emit(f"result={'PASS' if exit_code == 0 else 'FAIL'} (exit_code={exit_code})")
    if test_summary:
        emit(f"tests={test_summary}")
    if build_summary:
        emit(f"build={build_summary}")
    if error_line and (exit_code is None or exit_code != 0):
        emit(f"error_hint={_truncate_text(error_line, max_chars=220)}")


def _print_node_update(node_name: str, payload: dict, emit: Callable[[str], None]) -> None:
    emit(f"\n=== Node: {node_name} ===")

    if "step_count" in payload:
        emit(f"step_count={payload['step_count']}")
    if "status" in payload:
        emit(f"status={payload['status']}")
    if "consecutive_errors" in payload:
        emit(f"consecutive_errors={payload['consecutive_errors']}")

    messages = payload.get("messages", [])
    if isinstance(messages, list) and messages:
        latest = messages[-1]
        if isinstance(latest, ToolMessage):
            emit("tool_output:")
            _print_tool_result_summary(str(latest.content), emit)
            emit(_format_message_excerpt(latest))
        elif isinstance(latest, AIMessage):
            tool_calls = getattr(latest, "tool_calls", None)
            if tool_calls:
                emit("ai_tool_call:")
                emit(_truncate_text(str(tool_calls), max_chars=900))
            else:
                emit("ai_message:")
                emit(_format_message_excerpt(latest))
        else:
            emit("message:")
            emit(_format_message_excerpt(latest))

    if "summary_of_knowledge" in payload:
        emit("summary_of_knowledge:")
        emit(_truncate_text(str(payload["summary_of_knowledge"]), max_chars=700))


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = AgentConfig()
    if args.model is not None:
        config = replace(config, model_name=args.model)
    if args.max_steps is not None:
        config = replace(config, max_steps=args.max_steps)
    if args.repo_dir is not None:
        config = replace(config, repo_dir=Path(args.repo_dir))
    if args.clone_url is not None:
        config = replace(config, clone_url=args.clone_url)
    graph = build_graph(config)
    log_path = Path(args.log_file) if args.log_file else None
    log_handle = None

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    def emit(line: str) -> None:
        print(line)
        if log_handle is not None:
            log_handle.write(line + "\n")
            log_handle.flush()

    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "Run the full clone/explore/build/test workflow for the repository "
                    "and produce a final report."
                )
            )
        ],
        "summary_of_knowledge": "",
        "step_count": 0,
        "consecutive_errors": 0,
        "status": "EXPLORING",
    }

    try:
        recursion_limit = max(50, config.max_steps * 4 + 10)
        if not args.verbose_loop:
            result = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
            last_message = result["messages"][-1]
            emit(str(last_message.content))
            return

        final_report = None
        for event in graph.stream(initial_state, config={"recursion_limit": recursion_limit}, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, payload in event.items():
                if isinstance(payload, dict):
                    _print_node_update(node_name, payload, emit)
                    if node_name == "generate_report":
                        report_messages = payload.get("messages", [])
                        if isinstance(report_messages, list) and report_messages:
                            final_report = str(report_messages[-1].content)

        emit("\n=== Final Report ===")
        emit(final_report or "<no final report captured>")
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
