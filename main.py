from __future__ import annotations

import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.config import AgentConfig
from agent.graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-constrained build agent")
    parser.add_argument("--model", default="gpt-5.3-codex", help="LLM model name")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum reasoning loops")
    parser.add_argument("--repo-dir", default="workspace", help="Local workspace directory")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = AgentConfig(
        model_name=args.model,
        max_steps=args.max_steps,
        repo_dir=Path(args.repo_dir),
    )
    graph = build_graph(config)

    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "Run the full clone/explore/build/test workflow for nlohmann/json "
                    "and produce a final report."
                )
            )
        ],
        "summary_of_knowledge": "",
        "step_count": 0,
        "consecutive_errors": 0,
        "status": "EXPLORING",
    }

    result = graph.invoke(initial_state)
    last_message = result["messages"][-1]
    print(str(last_message.content))


if __name__ == "__main__":
    main()
