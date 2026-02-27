# Detailed Documentation: Context-Constrained LangGraph Build Agent

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Purpose & Problem Statement](#2-project-purpose--problem-statement)
3. [High-Level Architecture](#3-high-level-architecture)
4. [State Schema (`AgentState`)](#4-state-schema-agentstate)
5. [Configuration System (`AgentConfig`)](#5-configuration-system-agentconfig)
6. [Graph Construction & Edge Routing](#6-graph-construction--edge-routing)
7. [Node-by-Node Deep Dive](#7-node-by-node-deep-dive)
   - 7.1 [initialize_workspace](#71-initialize_workspace)
   - 7.2 [agent_reasoner](#72-agent_reasoner)
   - 7.3 [execute_tools (ToolNode)](#73-execute_tools-toolnode)
   - 7.4 [context_manager](#74-context_manager)
   - 7.5 [generate_report](#75-generate_report)
8. [Tool Specifications & Constraints](#8-tool-specifications--constraints)
9. [Context Management Policy (The Forget Policy)](#9-context-management-policy-the-forget-policy)
   - 9.1 [Token Budget Architecture](#91-token-budget-architecture)
   - 9.2 [Pruning Strategy – What Gets Forgotten](#92-pruning-strategy--what-gets-forgotten)
   - 9.3 [Protection Policy – What Is Preserved](#93-protection-policy--what-is-preserved)
   - 9.4 [Summary Breadcrumb System](#94-summary-breadcrumb-system)
   - 9.5 [Double-Fit Strategy](#95-double-fit-strategy)
   - 9.6 [Message Sanitization](#96-message-sanitization)
10. [Token Estimation System](#10-token-estimation-system)
11. [Error Tracking & Retry Policy](#11-error-tracking--retry-policy)
12. [Status Inference Policy](#12-status-inference-policy)
13. [Routing & Termination Policy](#13-routing--termination-policy)
14. [Report Generation Policy](#14-report-generation-policy)
15. [System Prompts & Agent Behavior](#15-system-prompts--agent-behavior)
16. [Entry Point & Runtime (`main.py`)](#16-entry-point--runtime-mainpy)
17. [File-by-File Reference](#17-file-by-file-reference)
18. [Data Flow Walkthrough (Full Cycle)](#18-data-flow-walkthrough-full-cycle)
19. [Design Decisions & Trade-offs](#19-design-decisions--trade-offs)

---

## 1. Executive Summary

This project implements a **ReAct-style autonomous build agent** using the [LangGraph](https://github.com/langchain-ai/langgraph) framework. The agent's job is to clone the `nlohmann/json` C++ repository, explore its structure, compile it, run its test suite, and produce a diagnostic report—all while **never exceeding 5,000 tokens** in its LLM context window.

The core innovation is a **custom context management node** (`context_manager`) that acts as a "forget policy" engine. After each tool execution it:
- Measures total token usage across all messages
- Evicts the oldest, least-important tool call/response pairs when the budget is exceeded
- Compresses evicted information into a running `summary_of_knowledge` string (breadcrumbs)
- Protects high-value messages (build failures, test results) from premature eviction

This creates a sliding-window memory system where the agent always has access to recent context plus a compressed history of everything it has already done.

---

## 2. Project Purpose & Problem Statement

**Goal:** Build an agent that can take a large C++ project (~40+ files), understand it, compile it, run tests, and explain failures—without being able to see the entire codebase at once.

**Hard constraint:** The LLM context window is capped at **5,000 tokens total** (system prompt + history + tool results + generation output). This forces the agent to implement its own memory management.

**What the agent must do:**
1. **Explore** — discover build system, directory structure, dependencies
2. **Execute** — run `cmake`, `make`/`ninja`, `ctest` commands
3. **Handle failures** — parse error output, locate relevant source files, iterate
4. **Report** — produce a clear summary of what passed, what failed, and why

**What the agent must NOT do:**
- Modify the target C++ source code
- Read entire files (must use line ranges)
- Recursively list the entire repository tree
- Exceed the 5,000-token budget

---

## 3. High-Level Architecture

The system is a **directed cyclic graph** with 5 nodes and conditional routing:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   START                                                     │
│     │                                                       │
│     ▼                                                       │
│   initialize_workspace                                      │
│     │                                                       │
│     ▼                                                       │
│   ┌─────────────────┐                                       │
│   │ agent_reasoner   │◄─────────────────────────────────┐   │
│   └────────┬────────┘                                   │   │
│            │                                            │   │
│     ┌──────┴──────┐                                     │   │
│     │  Conditional │                                    │   │
│     │    Route     │                                    │   │
│     └──┬───────┬──┘                                     │   │
│        │       │                                        │   │
│   tool_call  no_tool_call / step_limit                  │   │
│        │       │                                        │   │
│        ▼       ▼                                        │   │
│   execute_tools  generate_report ──► END                │   │
│        │                                                │   │
│        ▼                                                │   │
│   context_manager ──────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The **ReAct loop** is: `agent_reasoner → execute_tools → context_manager → agent_reasoner` (repeat).

---

## 4. State Schema (`AgentState`)

Defined in `agent/state.py`:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary_of_knowledge: str
    step_count: int
    consecutive_errors: int
    status: Status  # Literal["EXPLORING", "BUILDING", "TESTING", "FAILED", "SUCCESS"]
```

### Field Details

| Field | Type | Purpose |
|---|---|---|
| `messages` | `list[BaseMessage]` | The conversation history (Human, AI, Tool messages). Uses LangGraph's `add_messages` reducer which merges new messages into the existing list by default. |
| `summary_of_knowledge` | `str` | A running plain-text summary of everything the agent has learned, including compressed information from evicted messages. Acts as long-term memory. |
| `step_count` | `int` | Counts how many times `agent_reasoner` has executed. Hard-capped at 15 to prevent infinite loops. Incremented by 1 each time `agent_reasoner` runs. |
| `consecutive_errors` | `int` | Tracks how many consecutive tool executions returned the **same** error signature. Used to detect "stuck" loops. Resets to 0 when a different result appears. |
| `status` | `Status` | Current phase of execution. Transitions are: `EXPLORING → BUILDING → TESTING → SUCCESS/FAILED`. Inferred automatically from command output. |

### The `add_messages` Reducer

The `messages` field uses LangGraph's built-in `add_messages` annotation. When a node returns `{"messages": [new_msg]}`, the reducer **appends** the new message to the existing list rather than replacing it. This is critical because nodes like `context_manager` return the **full pruned list** (replacing entirely) while `agent_reasoner` returns only the new AI response (appending).

**Important exception:** When `context_manager` returns the entire pruned message list, the replacement happens because the context manager manipulates the full list and returns it as the new state value. LangGraph's `add_messages` reducer handles this via message ID matching—messages with existing IDs are updated in place, and new messages are appended.

---

## 5. Configuration System (`AgentConfig`)

Defined in `agent/config.py` as a frozen dataclass:

```python
@dataclass(frozen=True)
class AgentConfig:
    model_name: str          = "gpt-5.3-codex"
    fallback_chat_model: str = "gpt-4o-mini"
    max_steps: int           = 15
    token_budget: int        = 5000
    prune_threshold: int     = 4000
    output_token_budget: int = 1000
    input_token_budget: int  = 4000
    failure_retry_limit: int = 2
    repo_dir: Path           = Path("workspace")
    clone_url: str           = "https://github.com/nlohmann/json"
```

### Token Budget Breakdown

```
Total Budget:  5,000 tokens
├── Input Budget:  4,000 tokens  (system + summary + history sent TO the LLM)
├── Output Budget: 1,000 tokens  (max tokens the LLM can GENERATE)
└── Prune Threshold: 4,000 tokens (context_manager triggers pruning above this)
```

### Validation Invariants (enforced in `__post_init__`)

1. `max_steps` must be ≤ 15 — prevents runaway loops
2. `input_token_budget + output_token_budget` must be ≤ `token_budget` — ensures the total never exceeds 5,000
3. `token_budget` must be exactly 5,000 — per specification, this cannot be changed

### Environment Variable Overrides

Every field reads from environment variables (`AGENT_MODEL`, `AGENT_MAX_STEPS`, etc.), allowing configuration without code changes. This is useful for different deployment environments or testing scenarios.

---

## 6. Graph Construction & Edge Routing

Defined in `agent/graph.py`:

```python
def build_graph(config: AgentConfig):
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("initialize_workspace", ...)
    graph.add_node("agent_reasoner", ...)
    graph.add_node("execute_tools", ToolNode(ALL_TOOLS))
    graph.add_node("context_manager", ...)
    graph.add_node("generate_report", ...)

    # Edges
    graph.add_edge(START, "initialize_workspace")
    graph.add_edge("initialize_workspace", "agent_reasoner")
    graph.add_conditional_edges("agent_reasoner", route_from_reasoner, {...})
    graph.add_edge("execute_tools", "context_manager")
    graph.add_edge("context_manager", "agent_reasoner")
    graph.add_edge("generate_report", END)

    return graph.compile()
```

### Edge Table

| From | To | Condition |
|---|---|---|
| `START` | `initialize_workspace` | Always (entry point) |
| `initialize_workspace` | `agent_reasoner` | Always |
| `agent_reasoner` | `execute_tools` | If last AI message contains tool calls |
| `agent_reasoner` | `generate_report` | If no tool calls OR `step_count >= max_steps` |
| `execute_tools` | `context_manager` | Always |
| `context_manager` | `agent_reasoner` | Always (back-loop) |
| `generate_report` | `END` | Always (terminal) |

### Config Injection

Each node is wrapped in a lambda that captures the `AgentConfig` instance: `lambda state: node_fn(state, config)`. This means configuration is fixed at graph construction time and shared across all nodes.

---

## 7. Node-by-Node Deep Dive

### 7.1 `initialize_workspace`

**Location:** `agent/nodes.py` → `initialize_workspace()`

**Purpose:** Clone the `nlohmann/json` repository and set the process working directory.

**Behavior:**
1. Creates the workspace directory (`workspace/`) if it doesn't exist
2. Checks if `workspace/json/.git` already exists (idempotent)
3. If not cloned, runs `git clone https://github.com/nlohmann/json` via `subprocess.run`
4. Changes the **process working directory** (`os.chdir`) to `workspace/json`
5. Produces a `HumanMessage` instructing the agent to start exploring
6. Updates `summary_of_knowledge` with initialization status
7. Sets `status` to `"EXPLORING"`

**Returns:**
```python
{
    "messages": [HumanMessage(content="Start exploring and building...")],
    "summary_of_knowledge": "<updated summary>",
    "status": "EXPLORING",
}
```

**Key design note:** The `os.chdir()` call means all subsequent tool calls (`execute_shell_command`, `list_directory`, etc.) run from within the cloned repository. This is critical—tools use `Path.cwd()` as their root.

---

### 7.2 `agent_reasoner`

**Location:** `agent/nodes.py` → `agent_reasoner()`

**Purpose:** The LLM "brain" that decides what to do next. This is where the model reasons about the current state and emits either a tool call or a final answer.

**Step-by-step execution:**

1. **Calculate output budget:** `max(256, min(700, output_token_budget // 2))` — the reasoner gets at most 700 tokens for its response (half the total output budget), ensuring the report node has tokens left.

2. **Build the LLM instance:** Creates a `ChatOpenAI` model with `temperature=0` (deterministic) and binds all 4 tools to it. For Codex/GPT-5 models, it enables the Responses API.

3. **Construct the message payload:**
   - `SystemMessage` — the reasoner system prompt with rules
   - `HumanMessage` — the current `summary_of_knowledge`
   - The recent message `history` (pruned to fit the input budget)

4. **Fit messages to budget:** Calls `_fit_reasoner_messages_to_budget()` which iteratively drops the oldest tool/observation pairs until the total fits within `input_token_budget` (4,000 tokens). See [Section 9.5](#95-double-fit-strategy).

5. **Sanitize message sequence:** Removes orphaned `ToolMessage`s that no longer have a corresponding `AIMessage` with a matching `tool_call_id`. See [Section 9.6](#96-message-sanitization).

6. **Invoke the model:** Sends the fitted messages to the LLM and gets a response.

7. **Force-retry check:** If the model decided to stop (no tool calls) but the agent is in a `FAILED` state and hasn't exhausted retry attempts, the response is **overridden** with a forced diagnostic `ctest` command. See [Section 11](#11-error-tracking--retry-policy).

8. **Return:** Appends the AI response to messages and increments `step_count`.

```python
return {
    "messages": [response],
    "step_count": state["step_count"] + 1,
}
```

---

### 7.3 `execute_tools` (ToolNode)

**Location:** `agent/graph.py` — `ToolNode(ALL_TOOLS)`

**Purpose:** LangGraph's built-in `ToolNode` that automatically executes the tool(s) requested by the most recent `AIMessage`'s `tool_calls`.

**Behavior:**
- Looks at the last `AIMessage` in the state's `messages`
- Extracts each `tool_call` (name + arguments)
- Invokes the corresponding tool function
- Returns `ToolMessage` objects with results
- Each `ToolMessage` carries the `tool_call_id` linking it back to the request

This node requires no custom code—it delegates to the tool functions defined in `agent/tools.py`.

---

### 7.4 `context_manager`

**Location:** `agent/nodes.py` → `context_manager()`

**Purpose:** The "secret weapon" — the custom memory management node that enforces the token budget after every tool execution. This is the **forget policy engine**.

**Full behavior (in order):**

1. **Copy messages:** Takes a mutable copy of the full message list.

2. **Estimate tokens:** Uses `estimate_token_count()` to measure the entire message list.

3. **Pruning loop:** While `token_count > prune_threshold` (4,000) AND there are more than 4 messages:
   - Identify **protected indexes** — messages containing important build/test output (see Section 9.3)
   - Try to pop the **oldest tool/observation pair** (an `AIMessage` with `tool_calls` plus its associated `ToolMessage`s), skipping protected pairs
   - If no pair is available, pop the **oldest non-protected message** of any type
   - If absolutely nothing can be removed (all protected), force-remove the first message
   - Re-estimate tokens after each removal
   - Collect all removed messages into a `dropped` list

4. **Sanitize:** Remove orphaned `ToolMessage`s left behind by selective pruning.

5. **Update summary:** If any messages were dropped:
   - Run `_summarize_messages()` on the evicted messages to extract structured breadcrumbs
   - Merge the breadcrumb into `summary_of_knowledge`
   - Cap the summary at `max(256, min(1200, input_token_budget // 3))` tokens

6. **Track errors:** Compute `consecutive_errors` by comparing the error signature of the last two `ToolMessage`s.

7. **Infer status:** Update `status` based on the most recent command output.

**Returns the full replacement state:**
```python
{
    "messages": messages,           # The pruned list
    "summary_of_knowledge": summary, # Updated breadcrumbs
    "consecutive_errors": consecutive_errors,
    "status": status,
}
```

> This node is the **only node** that returns the complete message list (replacement), while other nodes return only new messages (appending).

---

### 7.5 `generate_report`

**Location:** `agent/nodes.py` → `generate_report()`

**Purpose:** Produce the final human-readable report summarizing the build/test outcome.

**Step-by-step:**

1. **Calculate output budget:** `max(512, output_token_budget)` — the report gets the full 1,000-token output budget.

2. **Build an LLM instance** (no tools bound — this is pure generation).

3. **Collect environment facts:** Runs quick shell commands to gather:
   - Current working directory and path lengths
   - Python, CMake, g++, git versions
   - Git long-paths configuration
   - Path length risk assessment (>120 chars = high risk on Windows)

4. **Build CTest evidence snapshot:** Scans all retained `ToolMessage`s for `ctest` command outputs and extracts:
   - Full-suite run summaries (pass/fail totals)
   - Targeted re-run summaries
   - Failed test names and failure types (Failed vs. Timeout)
   - Consistency checks between full and targeted runs

5. **Construct the report request message** with all evidence.

6. **Fit to budget** using `_fit_report_messages_to_budget()`.

7. **Invoke the model** to generate the report.

---

## 8. Tool Specifications & Constraints

Defined in `agent/tools.py`. Four tools are available:

### 8.1 `execute_shell_command(cmd: str) -> str`

Runs any shell command via `subprocess.run(shell=True)`.

**Output format:**
```
[cmd]=<the command>
[exit_code]=<return code>
[stdout]
<stdout content>
[stderr]
<stderr content>
```

**Truncation policy:** If output exceeds 100 lines OR ~1,000 characters:
- Keep the first 50 lines
- Insert `... <output truncated> ...`
- Keep the last 50 lines

This is critical for C++ compilation errors, where template error cascades can produce thousands of lines but the root cause is typically at the top or bottom.

**Windows normalization:** On Windows (`os.name == "nt"`):
- Strips `./` prefixes from commands (Unix-ism)
- Converts forward-slash paths to backslash for `build/tests/` → `build\tests\`

### 8.2 `list_directory(path: str, depth: int = 1) -> str`

Lists directory contents with bounded recursion.

**Constraints:**
- `depth` must be 0–3 (prevents full recursive dumps)
- Sorts entries: directories first, then files, alphabetically
- Directories are suffixed with `/`

### 8.3 `read_file_chunk(filepath: str, start_line: int, end_line: int) -> str`

Reads a specific line range from a file.

**Constraints:**
- `start_line` must be > 0
- `end_line` must be ≥ `start_line`
- Maximum chunk size: 250 lines
- Lines are returned with their line numbers prefixed: `42: content`

The agent is **never allowed** to read entire files. It must obtain line numbers from error messages or `search_codebase` output first.

### 8.4 `search_codebase(regex_pattern: str) -> str`

Searches all text files in the repository with a regex pattern.

**Output format:** `relative/path:line_number:matching_line`

**Constraints:**
- Maximum 200 matches
- Excludes binary files (`.png`, `.jpg`, `.pdf`, `.zip`, `.exe`, `.dll`)
- Excludes directories: `.git`, `build`, `dist`, `out`, `node_modules`, `__pycache__`
- Returns `<no matches>` when nothing is found

---

## 9. Context Management Policy (The Forget Policy)

This is the most sophisticated part of the system. The forget policy operates at **three levels** to ensure the 5,000-token budget is never exceeded.

### 9.1 Token Budget Architecture

```
Total: 5,000 tokens
┌─────────────────────────────────┐
│ Input Budget: 4,000 tokens      │
│ ┌─────────────────────────────┐ │
│ │ System Prompt (~200 tokens) │ │
│ │ Summary of Knowledge        │ │
│ │   (up to ~1,200 tokens)     │ │
│ │ Message History             │ │
│ │   (remaining budget)        │ │
│ └─────────────────────────────┘ │
├─────────────────────────────────┤
│ Output Budget: 1,000 tokens     │
│ ┌─────────────────────────────┐ │
│ │ Reasoner output: ≤700 tok   │ │
│ │ Report output:  ≤1,000 tok  │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘

Prune Threshold: 4,000 tokens
(context_manager fires when messages exceed this)
```

### 9.2 Pruning Strategy – What Gets Forgotten

When total message tokens exceed the `prune_threshold` (4,000), the context manager enters a **pruning loop**:

**Priority of removal (lowest priority = removed first):**

1. **Oldest tool call/observation pairs** — An `AIMessage` with `tool_calls` plus its associated `ToolMessage`(s). These are whole "question + answer" units. Removing them together avoids orphaned messages.

2. **Oldest non-protected individual messages** — If no complete pairs can be removed, individual messages are popped starting from the oldest.

3. **Force removal** — If all messages are protected, the absolute oldest message is force-removed as a last resort.

**The pruning loop continues** until either:
- Token count drops below the threshold, OR
- Only 4 messages remain (safety floor to avoid empty context)

### 9.3 Protection Policy – What Is Preserved

Not all messages are equal. The `_important_message_indexes()` function identifies high-value messages that should be **kept as long as possible**:

A `ToolMessage` is marked as **important** if it matches ANY of these criteria:

| Criterion | Pattern | Why It Matters |
|---|---|---|
| Full CTest run | `ctest` command without `-R` flag | Contains the authoritative pass/fail totals |
| Test pass/fail summary | `\d+% tests passed, \d+ tests failed out of \d+` | Critical evidence for the report |
| Named test failures | `the following tests failed` | Lists which specific tests failed |
| Timeout | `***timeout` | Indicates a timeout failure vs. assertion failure |
| Assertion failure | `did not throw` or `is not correct` | Specific root cause evidence |
| Non-zero exit code | `[exit_code]=[1-9]\d*` | Any command that failed |
| Successful build | `cmake --build` with `[exit_code]=0` | Confirms the build succeeded |

When a `ToolMessage` is marked important, its **preceding `AIMessage`** (the tool call that generated it) is also protected. This ensures the pair stays together.

### 9.4 Summary Breadcrumb System

When messages are evicted, they are **not simply thrown away**. The `_summarize_messages()` function extracts structured breadcrumbs:

**Extraction pipeline for each evicted message:**

1. **Commands:** Extracts `[cmd]=...` lines → stored as `commands=cmake --build . ; ctest ...`
2. **Exit codes:** Extracts `[exit_code]=N` lines → stored as `results=[exit_code]=0`
3. **Test summaries:** Extracts `N% tests passed, M tests failed out of K` → stored as-is
4. **Build summaries:** Counts `Built target` occurrences → stored as `build=Built targets observed: 12`
5. **Error lines:** Extracts first line containing `error`, `failed`, or `fatal` → stored as `errors=...`
6. **File references:** Extracts file:line patterns (e.g., `foo.cpp:42`) → stored as `files=...`

**Output format:**
```
Pruned context summary: commands=cmake --build . | results=[exit_code]=0 | errors=... | files=...
```

**Summary capping:** The `summary_of_knowledge` string is itself capped at `max(256, min(1200, input_token_budget / 3))` tokens. When it grows too large, the oldest portion is truncated using the `trim_text_to_token_budget()` function (which iteratively cuts to 80% of its length until it fits).

### 9.5 Double-Fit Strategy

Context fitting happens at **two stages**:

1. **In `context_manager`** (after tool execution): Prunes the actual state's message list. This is the "between-turns" cleanup.

2. **In `agent_reasoner` / `generate_report`** (before LLM calls): Re-fits messages to the `input_token_budget` using `_fit_reasoner_messages_to_budget()` or `_fit_report_messages_to_budget()`. This is a "just-in-time" cleanup that handles edge cases where the state messages are still slightly over budget.

The just-in-time fitting uses the same strategy: drop oldest tool pairs, then oldest individual messages, then truncate the summary itself as a last resort.

### 9.6 Message Sanitization

After pruning, the `_sanitize_tool_message_sequence()` function ensures message sequence integrity:

**Problem:** If an `AIMessage` with tool calls is removed but its `ToolMessage` responses remain (or vice versa), the LLM will see orphaned messages and may get confused. Some LLM APIs will outright reject such sequences.

**Solution:** The sanitizer:
1. Scans all `AIMessage`s to build a set of valid `tool_call_id`s
2. Removes any `ToolMessage` whose `tool_call_id` is NOT in the valid set
3. Non-tool messages (Human, System) pass through unchanged

This runs after every pruning operation and before every LLM call.

---

## 10. Token Estimation System

Defined in `agent/token_utils.py`.

### Primary Method: `estimate_token_count(messages, model_name)`

Uses the `tiktoken` library to get **exact** token counts:
- Attempts to get the model-specific tokenizer via `tiktoken.encoding_for_model(model_name)`
- Falls back to `cl100k_base` encoding if the model isn't recognized
- Adds 4 tokens per message as overhead (for role markers, separators, etc.)

### Fallback Method

If `tiktoken` is not installed, uses a rough approximation: `total_characters / 4`.

### Text Trimming: `trim_text_to_token_budget(text, model_name, max_tokens)`

Iteratively shrinks text to fit a token budget:
- If already within budget, returns as-is
- Otherwise, repeatedly truncates to 80% of current length until it fits
- Minimum length: 32 characters (safety floor)

This is used for capping `summary_of_knowledge` and as a last resort for fitting messages.

---

## 11. Error Tracking & Retry Policy

### Consecutive Error Detection

The `_compute_consecutive_errors()` function tracks if the agent is stuck in a loop producing the same error:

1. Extracts an "error signature" from each `ToolMessage` — the first line containing `error`, `failed`, or `fatal`
2. Compares the last two tool messages' signatures
3. If they match, increments `consecutive_errors`
4. If they differ, resets to 1 (new error) or 0 (no error)

### Forced Retry Policy

When `agent_reasoner` produces a response with **no tool calls** (meaning it wants to stop), the `_should_force_failure_retry()` function checks:

| Condition | Required |
|---|---|
| Model output has no tool calls | Yes |
| Current status is `FAILED` | Yes |
| `step_count` < `max_steps - 1` | Yes (at least 1 step remaining) |
| `consecutive_errors` < `failure_retry_limit` (2) | Yes (not stuck) |

If **all conditions** are met, the model's response is **overridden** with a forced diagnostic command:

- **First retry** (`consecutive_errors == 0`): Runs `ctest --test-dir build --output-on-failure -j1` — a full test suite re-run with detailed failure output
- **Subsequent retries**: Runs `ctest --test-dir build -R "fetch_content|regression1|testsuites|class_parser" --output-on-failure -V` — a targeted re-run of known problematic tests

This ensures the agent doesn't give up prematurely when failures are detected but not yet fully diagnosed.

---

## 12. Status Inference Policy

The `_infer_status()` function automatically determines the agent's current phase from the most recent tool output:

| Command Type | Exit Code | Resulting Status |
|---|---|---|
| `ctest` | 0 | `SUCCESS` |
| `ctest` | non-zero | `FAILED` |
| `cmake --build`, `ninja`, `make` | 0 | `BUILDING` |
| `cmake --build`, `ninja`, `make` | non-zero | `FAILED` |
| `cmake --build` (no exit code yet) | — | `BUILDING` |
| Any other command | 0 | unchanged |
| Any other command | non-zero | `FAILED` |
| Non-command output | — | unchanged |

**Detection method:**
- Checks for `[cmd]=` and `[exit_code]=` markers in the tool output
- Uses regex matching on the command string to classify type
- The status only transitions forward (exploring → building → testing → success/failed)

---

## 13. Routing & Termination Policy

Defined in `route_from_reasoner()`:

```python
def route_from_reasoner(state, config) -> str:
    if state["step_count"] >= config.max_steps:  # >= 15
        return "generate_report"
    last = _last_ai_message(state["messages"])
    if last and getattr(last, "tool_calls", None):
        return "execute_tools"
    return "generate_report"
```

### Decision Table

| Condition | Route |
|---|---|
| `step_count >= 15` | → `generate_report` (hard stop) |
| Last AI message has `tool_calls` | → `execute_tools` (continue loop) |
| Last AI message has no `tool_calls` | → `generate_report` (agent chose to stop) |

### Additional Safety: Recursion Limit

In `main.py`, the LangGraph runtime is configured with `recursion_limit = max(50, max_steps * 4 + 10)`. With `max_steps=15`, this is `max(50, 70) = 70`. This prevents graph cycles from running indefinitely even if step counting somehow fails.

---

## 14. Report Generation Policy

The report generation node uses a specialized system prompt (`REPORT_SYSTEM_PROMPT`) that requires:

1. **Build/test outcome** — clear pass/fail statement
2. **Full-suite CTest snapshot** — authoritative totals and failing test list
3. **Failure breakdown** — categorize as Failed vs. Timeout
4. **Consistency check** — compare full-suite results with targeted re-runs
5. **Key commands executed** — what was run
6. **Root cause analysis** — why failures occurred
7. **Environment sanity** — OS, toolchain versions, path length issues
8. **Local vs. upstream interpretation** — whether failures are environment-specific
9. **Suggested next steps** — what to do about failures

### Evidence Assembly

Before calling the LLM, the report node assembles two evidence blocks:

**Environment facts** (`_collect_environment_facts`):
```
cwd=C:\...\workspace\json
os_name=nt
cmake=cmake version 3.28.1
gxx=g++ (GCC) 13.1.0
path_length_risk=low
```

**CTest evidence snapshot** (`_build_ctest_evidence_snapshot`):
```
ctest_runs_observed=2
full_suite_cmd=ctest --test-dir build --output-on-failure
full_suite_summary=95% tests passed, 3 tests failed out of 60
full_suite_failed_tests=['test_regression', 'test_unicode']
targeted_cmd=ctest --test-dir build -R "regression" -V
targeted_summary=0% tests passed, 1 test failed out of 1
```

The report prompt explicitly instructs: **"Treat full-suite CTest evidence as authoritative when totals differ."**

---

## 15. System Prompts & Agent Behavior

Defined in `agent/prompts.py`:

### Reasoner Prompt

Gives the agent these behavioral rules:
1. Prefer short, targeted tool calls
2. Only use `read_file_chunk` with line ranges from error output (not arbitrary)
3. Do not recursively list the entire repository
4. Keep reasoning concise and action-oriented
5. Stop when enough evidence is gathered
6. On Windows, use Windows-safe paths (`build\\tests\\name.exe`, no `./` prefix)

### Report Prompt

Structures the final output with specific sections and rules:
- Full-suite results take precedence over targeted re-runs
- Must call out inconsistencies between full and targeted runs
- Must not claim success without command evidence

---

## 16. Entry Point & Runtime (`main.py`)

### Command-Line Interface

```bash
python main.py --model gpt-5.3-codex --max-steps 15 --repo-dir workspace --verbose-loop --log-file logs/run.log
```

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `gpt-5.3-codex` | LLM model name |
| `--max-steps` | `15` | Max reasoning loops |
| `--repo-dir` | `workspace` | Local workspace path |
| `--verbose-loop` | `false` | Print per-node updates during execution |
| `--log-file` | `""` | Optional file to write runtime logs |

### Execution Modes

**Normal mode** (no `--verbose-loop`):
- Calls `graph.invoke()` which runs the entire graph to completion
- Prints only the final AI message

**Verbose mode** (`--verbose-loop`):
- Uses `graph.stream(stream_mode="updates")` to get per-node state updates
- For each node update, prints:
  - Node name
  - `step_count`, `status`, `consecutive_errors`
  - Latest message excerpt (truncated to 900 chars)
  - For tool outputs: structured summaries (command, exit code, test results, error hints)
  - `summary_of_knowledge` (truncated to 700 chars)

### Initial State

```python
initial_state = {
    "messages": [HumanMessage(content="Run the full clone/explore/build/test workflow...")],
    "summary_of_knowledge": "",
    "step_count": 0,
    "consecutive_errors": 0,
    "status": "EXPLORING",
}
```

---

## 17. File-by-File Reference

| File | Lines | Purpose |
|---|---|---|
| `main.py` | 190 | CLI entry point, verbose streaming, log output |
| `agent/__init__.py` | 3 | Re-exports `build_graph` |
| `agent/config.py` | 28 | `AgentConfig` dataclass with validation |
| `agent/state.py` | 17 | `AgentState` TypedDict with `add_messages` reducer |
| `agent/graph.py` | 43 | LangGraph `StateGraph` construction and compilation |
| `agent/nodes.py` | 759 | All 5 node functions + 30+ helper functions |
| `agent/tools.py` | 131 | 4 tool definitions with constraints |
| `agent/token_utils.py` | 58 | Token counting and text trimming utilities |
| `agent/prompts.py` | 41 | System prompts for reasoner and report nodes |
| `requirements.txt` | 5 | Python dependencies |
| `ARCHITECTURE.md` | 86 | Spec compliance summary |
| `README.md` | 42 | Quick-start guide |

---

## 18. Data Flow Walkthrough (Full Cycle)

Here is an end-to-end trace of a single ReAct loop iteration:

### Step 0: Initialization
```
State: messages=[], summary="", step_count=0, status="EXPLORING"
  │
  ▼
initialize_workspace:
  - Clones repo (or detects it's already cloned)
  - os.chdir("workspace/json")
  - Returns: messages=[HumanMessage], summary="Workspace initialized...", status="EXPLORING"
```

### Step 1: First Reasoning
```
State: messages=[HumanMessage], summary="Workspace initialized...", step_count=0
  │
  ▼
agent_reasoner:
  - Builds: [SystemMessage, HumanMessage(summary), HumanMessage(original)]
  - Fits to 4,000 tokens (trivially fits at this point)
  - LLM decides: "I should list the directory first"
  - Returns: messages=[AIMessage(tool_calls=[list_directory(".", 2)])]
  - step_count → 1
```

### Step 2: Tool Execution
```
State: messages=[HumanMessage, AIMessage(tool_call)], step_count=1
  │
  ▼
execute_tools:
  - Runs list_directory(".", 2)
  - Returns: messages=[ToolMessage(content="CMakeLists.txt\nbuild/\ninclude/\n...")]
```

### Step 3: Context Management
```
State: messages=[HumanMessage, AIMessage, ToolMessage], step_count=1
  │
  ▼
context_manager:
  - Estimates tokens: ~800 (well under 4,000 threshold)
  - No pruning needed
  - No errors detected → consecutive_errors=0
  - Still EXPLORING
  - Returns state unchanged
```

### Step 4: Second Reasoning
```
State: messages=[HumanMessage, AIMessage, ToolMessage], step_count=1
  │
  ▼
agent_reasoner:
  - LLM sees directory listing, decides to read CMakeLists.txt
  - Returns: AIMessage(tool_calls=[read_file_chunk("CMakeLists.txt", 1, 50)])
  - step_count → 2
```

### ... (loop continues) ...

### Step N: Context Gets Full
```
State: messages=[many messages], token_count=4,500
  │
  ▼
context_manager:
  - token_count (4,500) > prune_threshold (4,000) → PRUNE!
  - Finds oldest non-protected tool pair: the list_directory call from step 1
  - Removes: AIMessage(list_directory) + ToolMessage(directory listing)
  - Summarizes: "Pruned context summary: commands=list_directory . | results=CMakeLists.txt present"
  - Appends to summary_of_knowledge
  - Re-checks: token_count now 3,200 → under threshold, stop pruning
  - Sanitizes message sequence
  - Returns pruned state
```

### Final Step: Report
```
State: status="FAILED", step_count=14
  │
  ▼
agent_reasoner:
  - LLM has no more ideas or decides it has enough evidence
  - Returns: AIMessage(content="Here is my analysis...") with NO tool_calls
  │
  ▼
route_from_reasoner:
  - No tool_calls → check force_retry
  - consecutive_errors >= failure_retry_limit → no force retry
  - Route to "generate_report"
  │
  ▼
generate_report:
  - Collects environment facts
  - Builds CTest evidence snapshot
  - Calls LLM with report prompt
  - Returns final report message
  │
  ▼
END
```

---

## 19. Design Decisions & Trade-offs

### Why a Custom Context Manager Instead of LangGraph's Built-in Trimming?

LangGraph provides a `messages_modifier` parameter and `trim_messages` utility, but these operate with simple sliding windows. The custom `context_manager` node provides:
- **Semantic-aware pruning** — protects important build/test evidence
- **Structured summarization** — extracts commands, errors, file references from evicted messages
- **Pair-wise removal** — keeps AIMessage↔ToolMessage pairs together
- **Status inference** — updates the agent's phase automatically

### Why `os.chdir()` Instead of Passing Paths?

The agent's tools use `Path.cwd()` as the implicit root. By changing the process working directory to the cloned repository, all subsequent tool calls automatically operate in the correct context without needing explicit path parameters. This mirrors how a human developer would `cd` into a project before working.

**Trade-off:** This mutates global process state, which could be problematic in multi-agent scenarios. Acceptable here since it's a single-agent system.

### Why Split Input/Output Budgets?

The 5,000-token budget is split 4,000/1,000 (input/output). This reserves enough generation capacity for the final report while maximizing the context available for reasoning. The reasoner is further limited to 700 tokens per response (half the output budget) to ensure multiple reasoning steps can occur within the overall budget.

### Why a Prune Threshold of 4,000?

The prune threshold equals the input budget. This means pruning starts exactly when messages would exceed what can be sent to the LLM. There's no "buffer zone" between the prune threshold and the input budget, which is aggressive but maximizes the context available to the agent at each step.

### Why Protect CTest Output?

In a build-and-test workflow, the test results are the **most valuable** information. A test summary line like `"95% tests passed, 3 tests failed out of 60"` is irreplaceable — it provides the basis for the entire final report. Protecting these messages ensures they survive pruning even as exploratory commands (directory listings, file reads) are evicted.

### Why Force Retries on Failure?

LLMs sometimes "give up" prematurely when they see a failure. The forced retry policy ensures:
1. At least one diagnostic re-run happens before reporting
2. Targeted test re-runs can isolate specific failures
3. The agent doesn't report vague failures when more specific evidence is obtainable

The retry limit (2) prevents infinite retry loops.

### Why Summarize Rather Than Just Forget?

Pure eviction would cause the agent to lose track of what it has already tried. The breadcrumb system preserves signal (commands run, exit codes, error patterns) while discarding noise (full stdout/stderr). This gives the agent just enough memory to avoid repeating actions and to maintain a coherent narrative in its final report.

---

*This document covers the complete codebase as of the current implementation. Every function, policy, and design decision in the LangGraph agent has been described.*
