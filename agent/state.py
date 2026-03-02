from __future__ import annotations

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


Status = Literal["EXPLORING", "CONFIGURING", "BUILDING", "TESTING", "FAILED", "SUCCESS"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary_of_knowledge: str
    step_count: int
    consecutive_errors: int
    status: Status
