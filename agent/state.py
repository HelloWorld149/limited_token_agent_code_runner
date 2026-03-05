from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------
Intent = Literal["QUESTION", "COMPILE", "RUN", "EXPLORE", "EXIT"]

# ---------------------------------------------------------------------------
# Build lifecycle state
# ---------------------------------------------------------------------------
BuildStatus = Literal[
    "IDLE", "CONFIGURING", "BUILDING", "TESTING", "FAILED", "SUCCESS"
]


@dataclass
class BuildState:
    """Mutable record tracking build lifecycle across turns."""

    status: BuildStatus = "IDLE"
    configured: bool = False
    built: bool = False
    tested: bool = False
    last_exit_code: int | None = None
    last_error: str = ""
    consecutive_errors: int = 0


# ---------------------------------------------------------------------------
# Codebase index (in-memory, NOT sent to LLM)
# ---------------------------------------------------------------------------
@dataclass
class FileEntry:
    path: str
    language: str
    size: int
    summary: str  # first-line / short description
    purpose: str = ""  # computed: module type, key exports, architectural role
    declarations: list[str] = field(default_factory=list)  # top-level class/function/module declarations


@dataclass
class SymbolEntry:
    name: str
    kind: str  # function | class | struct | macro
    file: str
    line: int


@dataclass
class CodebaseIndex:
    """Lightweight index built at startup for retrieval."""

    root: str = ""
    files: list[FileEntry] = field(default_factory=list)
    symbols: list[SymbolEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary_of_knowledge: str
    codebase_index: CodebaseIndex
    current_intent: Intent
    build_state: BuildState
    turn_count: int
    last_user_input: str
    _retrieved_context: str  # transient: set by retrieve_context, read by LLM nodes
