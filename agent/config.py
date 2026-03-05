from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for the conversational agent.

    The 5000-token budget is a hard invariant:
        input_token_budget + output_token_budget == token_budget
    """

    model_name: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "gpt-5.3-codex")
    )
    classifier_model: str = field(
        default_factory=lambda: os.getenv("AGENT_CLASSIFIER_MODEL", "gpt-4o-mini")
    )
    token_budget: int = field(
        default_factory=lambda: int(os.getenv("AGENT_TOKEN_BUDGET", "5000"))
    )
    input_token_budget: int = field(
        default_factory=lambda: int(os.getenv("AGENT_INPUT_TOKENS", "4000"))
    )
    output_token_budget: int = field(
        default_factory=lambda: int(os.getenv("AGENT_OUTPUT_TOKENS", "1000"))
    )
    prune_threshold: int = field(
        default_factory=lambda: int(os.getenv("AGENT_PRUNE_THRESHOLD", "3800"))
    )
    failure_retry_limit: int = field(
        default_factory=lambda: int(os.getenv("AGENT_FAILURE_RETRY_LIMIT", "3"))
    )
    max_tool_iterations: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_ITERATIONS", "3"))
    )
    workspace_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("AGENT_WORKSPACE_PATH", "workspace/json")
        )
    )
    # --- Subagent configuration ---
    subagent_model: str = field(
        default_factory=lambda: os.getenv("AGENT_SUBAGENT_MODEL", "gpt-5.3-codex")
    )
    use_retrieval_subagent: bool = field(
        default_factory=lambda: os.getenv("AGENT_USE_RETRIEVAL_SUBAGENT", "true").lower() == "true"
    )
    use_tool_summarizer: bool = field(
        default_factory=lambda: os.getenv("AGENT_USE_TOOL_SUMMARIZER", "true").lower() == "true"
    )
    use_conversation_compressor: bool = field(
        default_factory=lambda: os.getenv("AGENT_USE_CONVERSATION_COMPRESSOR", "true").lower() == "true"
    )
    use_multi_hop: bool = field(
        default_factory=lambda: os.getenv("AGENT_USE_MULTI_HOP", "true").lower() == "true"
    )
    retrieval_digest_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RETRIEVAL_DIGEST_TOKENS", "400"))
    )
    tool_summary_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_TOOL_SUMMARY_TOKENS", "200"))
    )

    def __post_init__(self) -> None:
        if self.input_token_budget + self.output_token_budget > self.token_budget:
            raise ValueError(
                "input_token_budget + output_token_budget must be <= token_budget"
            )
        if self.token_budget != 5000:
            raise ValueError("token_budget must remain 5000 per specification")
        if self.retrieval_digest_tokens >= self.token_budget:
            raise ValueError("retrieval_digest_tokens must be < token_budget (5000)")
        if self.tool_summary_tokens >= self.token_budget:
            raise ValueError("tool_summary_tokens must be < token_budget (5000)")
