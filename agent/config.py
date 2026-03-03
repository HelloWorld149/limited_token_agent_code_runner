from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    model_name: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "gpt-5.3-codex"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "25")))
    token_budget: int = field(default_factory=lambda: int(os.getenv("AGENT_TOKEN_BUDGET", "5000")))
    prune_threshold: int = field(default_factory=lambda: int(os.getenv("AGENT_PRUNE_THRESHOLD", "4000")))
    output_token_budget: int = field(default_factory=lambda: int(os.getenv("AGENT_OUTPUT_TOKENS", "1000")))
    input_token_budget: int = field(default_factory=lambda: int(os.getenv("AGENT_INPUT_TOKENS", "4000")))
    failure_retry_limit: int = field(default_factory=lambda: int(os.getenv("AGENT_FAILURE_RETRY_LIMIT", "3")))
    repo_dir: Path = field(default_factory=lambda: Path(os.getenv("AGENT_REPO_DIR", "workspace")))
    clone_url: str = field(default_factory=lambda: os.getenv("AGENT_CLONE_URL", "https://github.com/nlohmann/json"))

    def __post_init__(self) -> None:
        """Execute method `__post_init__` on `AgentConfig`.

        This routine is part of the agent workflow and keeps its existing runtime behavior.

        Returns:
            None: Result produced by this routine.
        """
        if self.max_steps > 50:
            raise ValueError("max_steps must be <= 50")
        if self.input_token_budget + self.output_token_budget > self.token_budget:
            raise ValueError("input_token_budget + output_token_budget must be <= token_budget")
        if self.token_budget != 5000:
            raise ValueError("token_budget must remain 5000 per specification")
        if self._report_output_budget > self.output_token_budget:
            raise ValueError("report_output_budget exceeds output_token_budget")

    @property
    def _report_output_budget(self) -> int:
        """Execute method `_report_output_budget` on `AgentConfig`.

        This routine is part of the agent workflow and keeps its existing runtime behavior.

        Returns:
            int: Result produced by this routine.
        """
        return min(self.output_token_budget, max(512, self.output_token_budget))
