from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AgentConfig:
    model_name: str = os.getenv("AGENT_MODEL", "gpt-5.3-codex")
    fallback_chat_model: str = os.getenv("AGENT_FALLBACK_CHAT_MODEL", "gpt-4o-mini")
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "15"))
    token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "5000"))
    prune_threshold: int = int(os.getenv("AGENT_PRUNE_THRESHOLD", "4000"))
    output_token_budget: int = int(os.getenv("AGENT_OUTPUT_TOKENS", "1000"))
    input_token_budget: int = int(os.getenv("AGENT_INPUT_TOKENS", "4000"))
    repo_dir: Path = Path(os.getenv("AGENT_REPO_DIR", "workspace"))
    clone_url: str = "https://github.com/nlohmann/json"

    def __post_init__(self) -> None:
        if self.max_steps > 15:
            raise ValueError("max_steps must be <= 15")
        if self.input_token_budget + self.output_token_budget > self.token_budget:
            raise ValueError("input_token_budget + output_token_budget must be <= token_budget")
        if self.token_budget != 5000:
            raise ValueError("token_budget must remain 5000 per specification")
