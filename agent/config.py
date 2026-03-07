from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


_OUTPUT_HEADROOM_CAP = 800
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    max_tool_iterations: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_ITERATIONS", "10"))
    )
    workspace_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("AGENT_WORKSPACE_PATH", "workspace/json")
        )
    )
    cache_directory: Path = field(
        default_factory=lambda: Path(
            os.getenv("AGENT_CACHE_DIRECTORY", str(_PROJECT_ROOT / ".cache"))
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
    index_cache_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_INDEX_CACHE_ENABLED", "false").lower() == "true"
    )
    use_embedding_retrieval: bool = field(
        default_factory=lambda: os.getenv("AGENT_USE_EMBEDDING_RETRIEVAL", "true").lower() == "true"
    )
    embedding_provider: str = field(
        default_factory=lambda: (os.getenv("AGENT_EMBEDDING_PROVIDER", "openai") or "openai").strip().lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("AGENT_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("AGENT_EMBEDDING_DIMENSIONS", "256"))
    )
    background_reindex_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_BACKGROUND_REINDEX_ENABLED", "true").lower() == "true"
    )
    background_reindex_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("AGENT_BACKGROUND_REINDEX_INTERVAL_SECONDS", "180"))
    )
    shell_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("AGENT_SHELL_TIMEOUT_SECONDS", "1500"))
    )
    allow_dangerous_shell_commands: bool = field(
        default_factory=lambda: os.getenv("AGENT_ALLOW_DANGEROUS_SHELL_COMMANDS", "false").lower() == "true"
    )

    def __post_init__(self) -> None:
        workspace_path = self.workspace_path.resolve()
        cache_directory = self.cache_directory.resolve()
        object.__setattr__(self, "workspace_path", workspace_path)
        object.__setattr__(self, "cache_directory", cache_directory)

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
        if self.embedding_provider not in {"auto", "openai", "hashing"}:
            raise ValueError("embedding_provider must be one of: auto, openai, hashing")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be > 0")
        if self.background_reindex_interval_seconds <= 0:
            raise ValueError("background_reindex_interval_seconds must be > 0")
        if self.shell_timeout_seconds <= 0:
            raise ValueError("shell_timeout_seconds must be > 0")
        if cache_directory == workspace_path or cache_directory.is_relative_to(workspace_path):
            raise ValueError("cache_directory must be outside workspace_path")

    @property
    def effective_output_budget(self) -> int:
        """The actual max_tokens sent to the LLM, capped for formatting headroom.

        output_token_budget is nominally 1000 but we cap at _OUTPUT_HEADROOM_CAP
        (800) to leave room for response formatting overhead.
        """
        return min(self.output_token_budget, _OUTPUT_HEADROOM_CAP)
