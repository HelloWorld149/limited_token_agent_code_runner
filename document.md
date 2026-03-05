# DOCUMENTATION — Context-Constrained AI Codebase Assistant

> **Complete Technical Reference**
> Last updated: 2026-03-05

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
   - [High-Level Flow](#21-high-level-flow)
   - [Graph Topology](#22-graph-topology)
   - [Sub-Agent System](#23-sub-agent-system)
3. [Token Budget System](#3-token-budget-system)
   - [Hard Invariant](#31-hard-invariant)
   - [Per-Turn Budget Allocation](#32-per-turn-budget-allocation)
   - [Four Pillars of Context Management](#33-four-pillars-of-context-management)
4. [Module Reference](#4-module-reference)
   - [main.py](#41-mainpy)
   - [agent/config.py](#42-agentconfigpy)
   - [agent/state.py](#43-agentstatepy)
   - [agent/graph.py](#44-agentgraphpy)
   - [agent/nodes.py](#45-agentnodespy)
   - [agent/intent.py](#46-agentintentpy)
   - [agent/indexer.py](#47-agentindexerpy)
   - [agent/tools.py](#48-agenttoolspy)
   - [agent/prompts.py](#49-agentpromptspy)
   - [agent/subagents.py](#410-agentsubagentspy)
   - [agent/model_utils.py](#411-agentmodel_utilspy)
   - [agent/token_utils.py](#412-agenttoken_utilspy)
5. [Data Flow Walkthrough](#5-data-flow-walkthrough)
   - [Startup (Indexing)](#51-startup-indexing)
   - [Per-Turn Cycle](#52-per-turn-cycle)
   - [Three-Layer Retrieval Pipeline](#53-three-layer-retrieval-pipeline)
   - [ReAct Tool Loop](#54-react-tool-loop)
6. [State Machine & Data Structures](#6-state-machine--data-structures)
   - [AgentState (TypedDict)](#61-agentstate-typeddict)
   - [BuildState (Frozen Dataclass)](#62-buildstate-frozen-dataclass)
   - [CodebaseIndex, FileEntry, SymbolEntry](#63-codebaseindex-fileentry-symbolentry)
   - [Intent Types](#64-intent-types)
7. [Sub-Agent Architecture](#7-sub-agent-architecture)
   - [Retrieval Compressor](#71-retrieval-compressor)
   - [Tool Output Summarizer](#72-tool-output-summarizer)
   - [Conversation Compressor](#73-conversation-compressor)
   - [Multi-Hop Decomposer](#74-multi-hop-decomposer)
8. [Tool Reference](#8-tool-reference)
   - [execute_shell_command](#81-execute_shell_command)
   - [list_directory](#82-list_directory)
   - [read_file_chunk](#83-read_file_chunk)
   - [search_codebase](#84-search_codebase)
9. [Build State Machine](#9-build-state-machine)
10. [Configuration Reference](#10-configuration-reference)
11. [Test Suite](#11-test-suite)
12. [Coding Conventions](#12-coding-conventions)
13. [Dependencies](#13-dependencies)
14. [Important Rules & Invariants](#14-important-rules--invariants)

---

## 1. Project Overview

This project is a **human-interactive AI coding assistant** built with LangGraph that helps users understand, build, and test the **nlohmann/json** C++ library while operating under a strict **5000-token LLM context limit**.

### Assignment Context

This is a take-home assignment for an **AI Engineer — Founding Team** position. The core challenge is **intelligent management of limited context** (5000 tokens) to accurately answer codebase queries and execute build/test commands against a real-world C++ project.

### Target Codebase

- **nlohmann/json** — a widely used C++ JSON library with 40+ source files, comprehensive test suite, and CMake build system
- Pre-downloaded at `workspace/json/`
- The agent **never clones or downloads repositories** — it operates only on the local copy
- Because the codebase is large, the agent **cannot load all files into context simultaneously** — it must use selective retrieval

### Two Core Capabilities

1. **Codebase Understanding (Q&A)** — Answer questions about architecture, structure, functions, and classes. All answers grounded in retrieved code evidence.
2. **Build & Execution** — Execute local build commands (`cmake`, `make`, `ninja`), parse compiler output, run the test suite (`ctest`), and report results.

---

## 2. Architecture

### 2.1 High-Level Flow

```
Startup:  START → index_workspace → END

Per-turn (REPL loop):
  START → classify_and_prepare → retrieve_context → route_by_intent:
    ├── answer_question  ──┐
    ├── run_build        ──┤── route_after_llm → execute_tools → handle_tool_result
    ├── run_tests        ──┤                      → continue_or_respond (ReAct loop)
    ├── explore_codebase ──┘
    └── exit → handle_exit → END
```

The agent runs as a conversational REPL. At startup it indexes the workspace once. Then for each user turn it classifies intent, retrieves context, routes to the appropriate handler, and optionally loops through tools via a ReAct pattern.

### 2.2 Graph Topology

Two LangGraph `StateGraph` instances are compiled:

1. **Init Graph** (`build_init_graph`):
   - Single node: `index_workspace`
   - Verifies workspace exists, builds codebase index, probes environment
   - Runs once at startup

2. **Turn Graph** (`build_turn_graph`):
   - Nodes: `classify_and_prepare`, `retrieve_context`, `answer_question`, `run_build`, `run_tests`, `explore_codebase`, `execute_tools` (ToolNode), `handle_tool_result`, `continue_or_respond`
   - Conditional edges route by intent and by LLM output (tool calls vs. final text)
   - The ReAct loop: `LLM node → route_after_llm → execute_tools → handle_tool_result → continue_or_respond → route_after_llm → ...`
   - All four intent handlers share the same ReAct tool loop via `route_after_llm`
   - `max_tool_iterations` (default 3) enforces a cap; once reached, `continue_or_respond` invokes the LLM without tool bindings to force a text response

### 2.3 Sub-Agent System

Sub-agents are **separate LLM calls** outside the main 5000-token budget. Each has its own 5000-token cap (`_SUBAGENT_TOKEN_CAP = 5000`). They expand effective capacity to ~20,000+ tokens per turn across multiple focused calls.

| Sub-Agent | Purpose | Input Budget | Output Budget |
|---|---|---|---|
| Retrieval Compressor | Raw code → dense digest | ~3500 tokens | ~400 tokens |
| Tool Output Summarizer | Build logs → compact summary | ~3500 tokens | ~200 tokens |
| Conversation Compressor | Rolling summary refresh | ~3000 tokens | ~400 tokens |
| Multi-Hop Decomposer | Complex Q → decompose → parallel → merge | Varies | ~500 tokens |

---

## 3. Token Budget System

### 3.1 Hard Invariant

```
input_token_budget (4000) + output_token_budget (1000) = token_budget (5000)
```

- Enforced in `AgentConfig.__post_init__()` — raises `ValueError` if violated
- `token_budget` must be exactly 5000 — validated at construction
- Output is further capped to 800 tokens via `effective_output_budget` property to leave formatting headroom

### 3.2 Per-Turn Budget Allocation

| Component | Tokens | Source |
|---|---|---|
| System prompt | ~150 | Intent-specific, selected from `INTENT_PROMPT_MAP` |
| Summary + retrieved context | ~600 | `summary_of_knowledge` + compressed code digest |
| Conversation history | ~1800 | Last 2–3 turns, older turns compressed |
| Tool schemas | ~300 | When tools are bound (reserved from `input_token_budget`) |
| User message | ~50–200 | Current turn input |
| **LLM output** | **≤800** | Capped via `effective_output_budget` |

### 3.3 Four Pillars of Context Management

1. **Selection** — Three-layer retrieval: path-aware direct load → keyword/symbol search → ReAct tool fallback
2. **Compression** — Sub-agents compress ~3000 tokens of raw code into ~400-token digests; tool outputs into ~200-token summaries
3. **Persistence** — Rolling `summary_of_knowledge` re-compressed every 3 turns via the conversation compressor sub-agent
4. **Eviction** — `fit_messages_to_budget()` drops oldest tool-call/observation pairs first, then oldest non-system messages; `sanitize_tool_message_sequence()` removes orphaned `ToolMessage`s

---

## 4. Module Reference

### 4.1 `main.py`

**REPL entry point** — input loop, graph invocation, output display.

- Loads `.env` via `python-dotenv`
- Validates workspace existence at `config.workspace_path`
- Runs the init graph once to index the workspace
- Enters a `while True` REPL loop:
  - Reads user input
  - Checks for exit commands
  - Appends `HumanMessage` to state
  - Invokes the turn graph with `recursion_limit=50`
  - Displays the response via `_display_response()`
- `_display_response()` walks messages in reverse to find the last `AIMessage`, handles Responses API list-of-blocks format, and prints trace logs (subagent count, debug)

### 4.2 `agent/config.py`

**`AgentConfig`** — immutable frozen dataclass with environment variable overrides.

**Fields:**

| Field | Type | Default | Env Var |
|---|---|---|---|
| `model_name` | `str` | `gpt-5.3-codex` | `AGENT_MODEL` |
| `classifier_model` | `str` | `gpt-4o-mini` | `AGENT_CLASSIFIER_MODEL` |
| `token_budget` | `int` | `5000` | `AGENT_TOKEN_BUDGET` |
| `input_token_budget` | `int` | `4000` | `AGENT_INPUT_TOKENS` |
| `output_token_budget` | `int` | `1000` | `AGENT_OUTPUT_TOKENS` |
| `max_tool_iterations` | `int` | `3` | `AGENT_MAX_TOOL_ITERATIONS` |
| `workspace_path` | `Path` | `workspace/json` | `AGENT_WORKSPACE_PATH` |
| `subagent_model` | `str` | `gpt-5.3-codex` | `AGENT_SUBAGENT_MODEL` |
| `use_retrieval_subagent` | `bool` | `true` | `AGENT_USE_RETRIEVAL_SUBAGENT` |
| `use_tool_summarizer` | `bool` | `true` | `AGENT_USE_TOOL_SUMMARIZER` |
| `use_conversation_compressor` | `bool` | `true` | `AGENT_USE_CONVERSATION_COMPRESSOR` |
| `use_multi_hop` | `bool` | `true` | `AGENT_USE_MULTI_HOP` |
| `retrieval_digest_tokens` | `int` | `400` | `AGENT_RETRIEVAL_DIGEST_TOKENS` |
| `tool_summary_tokens` | `int` | `200` | `AGENT_TOOL_SUMMARY_TOKENS` |

**Validation (`__post_init__`):**
- `input_token_budget + output_token_budget <= token_budget`
- `token_budget == 5000`
- `retrieval_digest_tokens < token_budget`
- `tool_summary_tokens < token_budget`

**Property:**
- `effective_output_budget` → `min(output_token_budget, 800)` — practical cap for LLM `max_tokens`

### 4.3 `agent/state.py`

**State definitions** for LangGraph and the build lifecycle.

**Types:**
- `Intent = Literal["QUESTION", "COMPILE", "RUN", "EXPLORE", "EXIT"]`
- `BuildStatus = Literal["IDLE", "CONFIGURING", "BUILDING", "TESTING", "FAILED", "SUCCESS"]`

**`BuildState`** — `@dataclass(frozen=True)`:

| Field | Type | Default |
|---|---|---|
| `status` | `BuildStatus` | `"IDLE"` |
| `configured` | `bool` | `False` |
| `built` | `bool` | `False` |
| `tested` | `bool` | `False` |
| `last_exit_code` | `int \| None` | `None` |
| `last_error` | `str` | `""` |
| `consecutive_errors` | `int` | `0` |

**`FileEntry`** — mutable dataclass:
- `path`, `language`, `size`, `summary`, `purpose`, `declarations`

**`SymbolEntry`** — mutable dataclass:
- `name`, `kind` (`function | class | struct | macro`), `file`, `line`

**`CodebaseIndex`** — mutable dataclass:
- `root`, `files: list[FileEntry]`, `symbols: list[SymbolEntry]`

**`AgentState`** — `TypedDict` with `add_messages` reducer:

| Key | Type | Description |
|---|---|---|
| `messages` | `Annotated[list[BaseMessage], add_messages]` | Conversation history with merge-by-ID reducer |
| `summary_of_knowledge` | `str` | Rolling compressed knowledge summary |
| `codebase_index` | `CodebaseIndex` | In-memory file/symbol index (not sent to LLM) |
| `current_intent` | `Intent` | Classified intent for current turn |
| `build_state` | `BuildState` | Build lifecycle tracker |
| `turn_count` | `int` | Monotonically increasing turn counter |
| `last_user_input` | `str` | Raw user input text |
| `_retrieved_context` | `str` | Compressed code context for current turn |
| `_tool_iteration_count` | `int` | ReAct loop counter (reset each turn) |
| `_turn_subagent_count` | `int` | Subagents used this turn |
| `_turn_debug_logs` | `list[str]` | Debug trace for this turn |

### 4.4 `agent/graph.py`

**LangGraph `StateGraph` definitions.**

**`build_init_graph(config) -> CompiledGraph`:**
- Single node: `index_workspace`
- `START → index_workspace → END`

**`build_turn_graph(config) -> CompiledGraph`:**
- Nodes: `classify_and_prepare`, `retrieve_context`, `answer_question`, `run_build`, `run_tests`, `explore_codebase`, `execute_tools` (ToolNode), `handle_tool_result`, `continue_or_respond`, `handle_exit`
- All node lambdas close over `config` and call the corresponding function from `nodes.py`
- Conditional edges:
  - `retrieve_context → route_by_intent` → one of: `answer_question`, `run_build`, `run_tests`, `explore_codebase`, or `handle_exit` (exit)
  - `handle_exit → END` (injects farewell AIMessage to prevent stale display)
  - `answer_question`, `run_build`, `run_tests`, `explore_codebase`, `continue_or_respond` → `route_after_llm` → either `execute_tools` or `END` (respond_to_user)
  - `execute_tools → handle_tool_result → continue_or_respond` (loops back)

### 4.5 `agent/nodes.py`

**Graph node functions, routers, LLM invocation, and helpers.** (783 lines)

**Nodes:**

| Function | Purpose |
|---|---|
| `index_workspace(state, config)` | Startup: verify workspace, `os.chdir()`, `set_workspace_root()`, `build_codebase_index()`, `_probe_environment()` |
| `classify_and_prepare(state, config)` | Single-pass context-aware intent classification via `classify_intent_sync()` (with dialog context), conversation compressor every 3 turns |
| `retrieve_context(state, config)` | Three-layer retrieval + subagent compression (retrieval or multi-hop) |
| `answer_question(state, config)` | QUESTION handler — calls `_invoke_llm_with_context(use_tools=True)` |
| `run_build(state, config)` | COMPILE handler — calls `_invoke_llm_with_context(use_tools=True)` |
| `run_tests(state, config)` | RUN handler — calls `_invoke_llm_with_context(use_tools=True)` |
| `explore_codebase(state, config)` | EXPLORE handler — calls `_invoke_llm_with_context(use_tools=True)` |
| `handle_tool_result(state, config)` | Post-process tool results: update `BuildState`, compress large `ToolMessage`s via tool summarizer subagent |
| `continue_or_respond(state, config)` | After tools, decide: more tools or text response. Enforces `max_tool_iterations` |

**Routers:**

| Function | Returns |
|---|---|
| `route_by_intent(state)` | `"answer_question"`, `"run_build"`, `"run_tests"`, `"explore_codebase"`, or `"exit"` |
| `route_after_llm(state)` | `"execute_tools"` (if last AIMessage has `tool_calls`) or `"respond_to_user"` |

**Key Internal Helpers:**

- `_invoke_llm_with_context(state, config, use_tools)` — Core LLM invocation:
  1. Selects intent-specific system prompt from `INTENT_PROMPT_MAP`
  2. Builds `SystemMessage` with `summary_of_knowledge` + `_retrieved_context`
  3. Assembles candidate messages: `[system, summary_msg, ...history]`
  4. Calls `fit_messages_to_budget()` (reserves 300 tokens for tool schemas if using tools)
  5. Calls `sanitize_tool_message_sequence()`
  6. Constructs `ChatOpenAI` via `build_chat_model()`
  7. Binds tools if `use_tools=True`
  8. Invokes model, normalizes response via `normalize_ai_message()`
  9. Falls back to error `AIMessage` on exception

- `_probe_environment(workspace_path)` — Detects OS, cmake, ninja, g++, make/mingw32-make, recommends generator
- `_update_build_state(messages, current)` — Parses recent `ToolMessage`s for `[cmd]=` and `[exit_code]=` markers, returns a new frozen `BuildState`

**retrieve_context Details:**

- **Layer 1 — Path-aware direct retrieval**: `detect_file_references()` + `detect_directory_references()` to find explicitly mentioned files. Reads up to 80 lines per file.
- **Layer 2 — Enhanced keyword + symbol search**: `search_index()` with fuzzy path matching. Reads ~45 lines around each match.
- **Fallback**: If nothing retrieved, injects `format_file_manifest_summary()`.
- **Subagent compression**: When `use_retrieval_subagent=True`, raw chunks are read with a ~3000-token budget and compressed to ~400 tokens. Complex questions use `multi_hop_decomposer_sync()` instead of the simple retrieval subagent.

### 4.6 `agent/intent.py`

**Intent classification** with async LLM + keyword fallback + follow-up classification.

**Public API:**

| Function | Description |
|---|---|
| `classify_intent_sync(user_input, model_name, *, previous_intent, last_ai_summary)` | Classify user intent via lightweight context-aware LLM (~200 tokens, separate from main budget). Receives optional dialog context (previous intent + last AI summary) to resolve ambiguous follow-ups in a single pass. Falls back to keyword heuristics on error. |
| `classify_followup_sync(user_input, previous_intent, last_ai_message, model_name)` | Legacy follow-up classifier (retained as utility). Classifies ambiguous follow-ups as `CONFIRM`, `CANCEL`, `EXIT`, or `NEW_REQUEST`. No longer used in the main flow — the context-aware `classify_intent_sync` handles this. |

**Internal:**

- `_fallback_classify(user_input)` — Keyword-based: `exit/quit/bye` → EXIT, `build/compile/cmake` → COMPILE, `test/run/ctest` → RUN, `list/search/find` → EXPLORE, else → QUESTION
- `_fallback_followup(user_input)` — Typo-tolerant via `difflib.get_close_matches()` with cutoff 0.82. Categories: confirm words (`yes`, `ok`, `proceed`), cancel words (`no`, `stop`, `cancel`), exit words (`exit`, `quit`, `bye`), else `NEW_REQUEST`
- `_normalize_text(text)` — Lowercase, collapse whitespace, strip non-alphanumeric
- `_fuzzy_contains(tokens, lexicon, cutoff)` — Check if any token is in or fuzzy-matches the lexicon

**Classifier System Prompt:** ~180 tokens, classifies into QUESTION / COMPILE / RUN / EXPLORE / EXIT. Includes instructions to use dialog context (previous intent + last assistant message summary) to resolve ambiguous follow-ups like "yes", "do it", "go ahead" — correctly mapping them to the offered action instead of misclassifying as EXIT.

**Follow-up System Prompt:** Returns exactly one token: CONFIRM / CANCEL / EXIT / NEW_REQUEST.

### 4.7 `agent/indexer.py`

**Codebase indexer** — file manifest, symbol table, purpose map, search. (492 lines)

**Public API:**

| Function | Description |
|---|---|
| `build_codebase_index(workspace_path)` | Walk workspace, build `CodebaseIndex` with files, symbols, and purpose/declaration metadata |
| `detect_file_references(user_input, index)` | Detect explicit file paths/names in user text and match to index entries |
| `detect_directory_references(user_input, index)` | Detect directory references and return all files in those directories |
| `search_index(index, query, max_results)` | Enhanced keyword + symbol search with fuzzy path matching and purpose-aware scoring |
| `format_file_manifest_summary(index, max_entries)` | Format compact manifest for LLM context injection |

**Constants:**
- `SKIP_DIRS`: `.git`, `build`, `build-mingw`, `build-ninja`, `__pycache__`, `node_modules`, `.venv`, `dist`, `out`
- `SKIP_EXTENSIONS`: `.png`, `.jpg`, `.jpeg`, `.gif`, `.pdf`, `.zip`, `.exe`, `.dll`, `.o`, `.obj`, `.a`, `.so`, `.dylib`, `.woff`, `.woff2`, `.ttf`, `.eot`, `.ico`
- `_LANG_MAP`: Extension → language mapping (`.h`/`.hpp`/`.cpp` → `c++`, `.py` → `python`, `.cmake` → `cmake`, etc.)

**Symbol Extraction (C/C++ only):**
- `_CPP_FUNCTION_RE` — Top-level function declarations
- `_CPP_CLASS_RE` — `class`/`struct` declarations (including templated)
- `_CPP_MACRO_RE` — `#define` macros (skips include guards and boilerplate)

**File Purpose Detection (`_detect_file_purpose`):**
- C++20 module interface/implementation (via `export module` / `module` patterns)
- Header file (via `#pragma once` or include guards)
- Executable entry point (via `int main(`)
- Test file (via `TEST_CASE` / `TEST_F` / `CATCH_TEST_CASE` patterns)
- CMake project root vs. build script
- Build configuration files (`Makefile`, `meson.build`, etc.)
- Documentation (markdown files)

**Rich Summary (`_build_rich_summary`):**
- First non-comment line
- `@brief` docstrings
- C++20 module exports
- Key `#include`s (first 5)
- Class/struct declarations (first 3)
- CMake project info

**Search Scoring (`search_index`):**
- Keyword match in path/summary/language/purpose: +1.0
- Keyword in path: +0.5 bonus
- Fuzzy path filename match: +4.0 (exact), +3.0 (substring), +2.0 (stem)
- Purpose match: +1.0
- Declaration match: +1.5
- Symbol exact name match: +2.0, in text: +1.5
- Results sorted by score descending, capped at `max_results`

### 4.8 `agent/tools.py`

**LangChain `@tool` definitions** — shell execution, file reading, directory listing, code search.

**Module-Level:**
- `_workspace_root: Path | None` — Set via `set_workspace_root()` at startup, used by `search_codebase`
- `get_workspace_root()` — Returns workspace root, falls back to `Path.cwd()`
- `_CRITICAL_PATTERNS` — Regex patterns for truncation-resistant output lines (test summaries, errors, CMake errors, build failures)
- `_CTEST_SUMMARY_RE` — Extracts `X% tests passed, Y tests failed out of Z`
- `_ERROR_HINT_PATTERNS` — Matches `error`, `fatal`, `failed`, `not recognized`

**Tools:**

See [Tool Reference](#8-tool-reference) for detailed documentation.

**Helper Functions:**
- `_truncate_output(text, max_chars=3000)` — Smart truncation: head 40 lines + critical pattern lines + tail 40 lines. Hard-truncates at `max_chars`.
- `_normalize_command_for_platform(cmd)` — On Windows: strips `./` prefixes, replaces `$(nproc)` → `%NUMBER_OF_PROCESSORS%`, `export` → `set`, `rm -rf` → `rmdir /s /q`, `rm -f` → `del /f`
- `_extract_ctest_summary(stdout, stderr)` — Parse CTest pass/fail counts
- `_extract_error_hint(stdout, stderr)` — Find first line with `error`/`fatal`/`failed`
- `_iter_text_files(root, include_build)` — Yield text files under root, respecting `SKIP_DIRS` and `SKIP_EXTENSIONS`

**`ALL_TOOLS`** list: `[execute_shell_command, list_directory, read_file_chunk, search_codebase]`

### 4.9 `agent/prompts.py`

**Intent-specific system prompts** selected by intent classification.

| Intent | Prompt | Focus |
|---|---|---|
| `QUESTION` | `QUESTION_SYSTEM_PROMPT` | Code Q&A: ground answers in context, use tools if needed, 3-8 sentences, reference file paths |
| `COMPILE` | `BUILD_SYSTEM_PROMPT` | Build: use cmake + detected tools, parse errors, suggest next steps, Windows-compatible |
| `RUN` | `TEST_SYSTEM_PROMPT` | Tests: use ctest, report pass/fail, identify failures, never modify source |
| `EXPLORE` | `EXPLORE_SYSTEM_PROMPT` | Navigation: list dirs, read files, search patterns, structured results |

**`INTENT_PROMPT_MAP`**: `dict[str, str]` mapping intent → system prompt.

### 4.10 `agent/subagents.py`

**Four sub-agent modules** running as independent LLM calls. (539 lines)

See [Sub-Agent Architecture](#7-sub-agent-architecture) for detailed documentation.

**Budget enforcement:**
- `_SUBAGENT_TOKEN_CAP = 5000` — Hard cap per sub-agent call
- `_MAX_QUERY_TOKENS = 300` — User queries trimmed before subagent processing
- `_enforce_budget(max_input, max_output)` — Clamps `input + output ≤ 5000`, shrinks input first

**Complexity Detection:**
- `is_complex_question(user_query)` — Heuristic: multiple `?`, comparison keywords ("compare", "difference between", "versus"), ≥2 file references, or long queries with commas/`and`

**All functions have async/sync pairs:**
- Async versions: `*_async()`
- Sync wrappers: `*_sync()` using `run_async()`

### 4.11 `agent/model_utils.py`

**Model type detection, ChatOpenAI construction, response normalization, async helpers.**

This is the **single source of truth** for model handling.

**`is_responses_model(model_name) -> bool`:**
- Chat-only patterns (checked first): `^gpt-4o`, `^gpt-4-`, `^gpt-3.5`, `^gpt-4\b`
- Responses API patterns: `\bcodex\b`, `\bo[134]\b`, `\bo[134]-`, `\bo[134]p\b`
- Uses compiled regexes with word boundaries for robust versioned name handling

**`build_chat_model(model_name, temperature, max_tokens) -> ChatOpenAI`:**
- Auto-detects Responses API via `is_responses_model()`
- Passes `use_responses_api=True` when appropriate
- **ALL code must use this function** instead of constructing `ChatOpenAI` directly

**`extract_text(content) -> str`:**
- Handles `str`, `list[dict]` (Responses API blocks with `"text"` key), `list[str]`, and other types
- Strips whitespace

**`normalize_ai_message(msg: AIMessage) -> AIMessage`:**
- Converts list-of-blocks content to plain string
- Preserves `tool_calls` and `id`

**`run_async(coro) -> Any`:**
- Handles three scenarios:
  1. No event loop → `asyncio.run()`
  2. Loop exists but not running → `loop.run_until_complete()`
  3. Loop already running → `ThreadPoolExecutor` + `asyncio.run()` (with 60s timeout)

### 4.12 `agent/token_utils.py`

**Token counting, text trimming, budget fitting, message sanitization.**

**Public API:**

| Function | Description |
|---|---|
| `estimate_token_count(messages, model_name)` | Total tokens for a message list (content + 4 overhead per message + tool_calls serialization) |
| `estimate_text_tokens(text, model_name)` | Tokens for a plain string |
| `trim_text_to_token_budget(text, model_name, max_tokens)` | O(log n) binary search trimming — finds longest prefix that fits within `max_tokens` |
| `fit_messages_to_budget(messages, model_name, input_budget)` | Drop oldest tool-observation pairs first, then oldest non-system messages |
| `sanitize_tool_message_sequence(messages)` | Remove orphaned `ToolMessage`s whose `tool_call_id` has no matching `AIMessage` |

**Internal:**
- `_get_encoder(model_name)` — `tiktoken.encoding_for_model()` with `cl100k_base` fallback
- `_pop_oldest_tool_observation_pair(messages)` — Removes first `AIMessage(tool_calls) + ToolMessage(s)` pair
- `_message_text(message)` — Extract text from `str`, `list`, or other content types

**Pruning Priority (in `fit_messages_to_budget`):**
1. Drop oldest AI(tool_calls) + ToolMessage pairs
2. Drop oldest non-system messages (preserves system prompt at index 0)
3. Stop when under budget or only 2 messages remain

---

## 5. Data Flow Walkthrough

### 5.1 Startup (Indexing)

```
main.py: main()
  ↓
  AgentConfig() constructed (env vars loaded)
  ↓
  Verify workspace/json/ exists
  ↓
  build_init_graph(config).invoke(init_state)
    ↓
    index_workspace():
      1. os.chdir(workspace_path)
      2. set_workspace_root(workspace_path)
      3. build_codebase_index(workspace_path)
         - Walk all files (skip SKIP_DIRS, SKIP_EXTENSIONS, >500KB)
         - For each file: detect language, build rich summary, detect purpose, extract declarations
         - For C/C++ files: extract symbols (classes, structs, macros, functions)
      4. _probe_environment(workspace_path)
         - Detect OS, cmake version, ninja version, g++ version, make/mingw32-make
         - Recommend cmake generator
      5. Return initial state:
         - summary_of_knowledge: "Workspace: ... | Files: N | Symbols: M | os=... | cmake=... | ..."
         - codebase_index: populated CodebaseIndex
         - build_state: BuildState(IDLE)
         - turn_count: 0
```

### 5.2 Per-Turn Cycle

```
User input → HumanMessage appended to state.messages
  ↓
turn_graph.invoke(state):
  ↓
  1. classify_and_prepare():
     a. classify_intent_sync(user_input, classifier_model, previous_intent=..., last_ai_summary=...)
        [single context-aware LLM call, ~200 tokens, resolves follow-ups in one pass]
     c. Every 3 turns: conversation_compressor_sync()  [sub-agent LLM call]
     d. Return: current_intent, turn_count++, reset _tool_iteration_count
  ↓
  2. retrieve_context():
     a. Layer 1: detect_file_references() + detect_directory_references() → direct load
     b. Layer 2: search_index() → keyword/symbol/fuzzy results → read code chunks
     c. Fallback: format_file_manifest_summary() if nothing found
     d. If use_retrieval_subagent:
        - Complex question? → multi_hop_decomposer_sync() [3-5 sub-agent LLM calls]
        - Simple? → retrieval_subagent_sync() [1 sub-agent LLM call]
     e. Return: _retrieved_context (compressed digest, ~400 tokens)
  ↓
  3. route_by_intent() → answer_question / run_build / run_tests / explore_codebase / handle_exit
  ↓
  4. Intent handler (all call _invoke_llm_with_context):
     a. Select intent-specific system prompt
     b. Build SystemMessage with summary + retrieved context
     c. Assemble [system, summary_msg, ...history]
     d. fit_messages_to_budget(effective_budget - 300 for tools)
     e. sanitize_tool_message_sequence()
     f. build_chat_model() → bind_tools(ALL_TOOLS) if use_tools
     g. model.invoke(candidate) → normalize_ai_message()
     h. Return: {"messages": [response]}
  ↓
  5. route_after_llm():
     - If last AIMessage has tool_calls → "execute_tools"
     - Else → "respond_to_user" (END)
  ↓
  6. [If tools called] ReAct loop:
     execute_tools (ToolNode) → handle_tool_result → continue_or_respond → route_after_llm
     - handle_tool_result: update BuildState, compress large outputs via tool summarizer
     - continue_or_respond: re-invoke LLM (without tools if max_iterations reached)
     - Loop until LLM produces text response or max_tool_iterations hit
  ↓
  7. Final AIMessage → displayed to user via _display_response()
```

### 5.3 Three-Layer Retrieval Pipeline

```
Layer 1: Path-Aware Direct Retrieval
  ├── detect_file_references() — regex match file paths/names in user input
  ├── detect_directory_references() — regex match directory paths
  ├── Match against CodebaseIndex entries
  └── Read up to 80 lines per file (30 if budget-constrained)

Layer 2: Enhanced Keyword + Symbol Search
  ├── search_index(index, user_input, max_results=10)
  │   ├── Score files: keyword in text (+1.0), in path (+0.5), filename match (+4.0)
  │   ├── Score symbols: keyword in text (+1.5), exact name (+2.0)
  │   └── Sort by score, take top results
  └── Read ~45 lines around each match (15 if budget-constrained)

Layer 3: ReAct Tool Fallback (handled by LLM)
  └── LLM can call read_file_chunk, list_directory, search_codebase on-demand
```

**Token budgets for raw code:**
- With retrieval subagent: read up to 3000 tokens of raw code (compressed to ~400)
- Without retrieval subagent: read up to 1800 tokens (injected directly)

### 5.4 ReAct Tool Loop

```
Intent handler → LLM produces AIMessage:
  ├── Has tool_calls? → route to execute_tools
  │     ├── ToolNode executes each tool, returns ToolMessages
  │     ├── handle_tool_result():
  │     │   ├── _tool_iteration_count++
  │     │   ├── _update_build_state() — parse [cmd]= / [exit_code]= markers
  │     │   └── If use_tool_summarizer: compress large ToolMessages
  │     └── continue_or_respond():
  │           ├── If _tool_iteration_count < max_tool_iterations:
  │           │   └── _invoke_llm_with_context(use_tools=True) — LLM can call more tools
  │           └── If _tool_iteration_count >= max_tool_iterations:
  │               └── _invoke_llm_with_context(use_tools=False) — force text response
  └── No tool_calls? → respond_to_user (END)
```

---

## 6. State Machine & Data Structures

### 6.1 AgentState (TypedDict)

The central state passed through all LangGraph nodes. Uses `add_messages` reducer for the `messages` field, which supports merge-by-ID semantics (critical for tool output compression — replacing `ToolMessage`s with compressed versions).

### 6.2 BuildState (Frozen Dataclass)

Immutable record tracking the build lifecycle. New instances are created via `BuildState(...)` constructor — never mutated. `_update_build_state()` in `nodes.py` constructs new instances based on ToolMessage parsing.

**State Transitions:**
```
IDLE ──cmake (success)──→ CONFIGURING
CONFIGURING ──make/ninja (success)──→ BUILDING
BUILDING ──ctest (success)──→ SUCCESS
Any state ──command fails──→ FAILED (consecutive_errors++)
FAILED ──command succeeds──→ (appropriate stage, consecutive_errors reset to 0)
```

### 6.3 CodebaseIndex, FileEntry, SymbolEntry

`CodebaseIndex` is built once at startup and **never sent to the LLM**. It serves as an in-memory search index for the retrieval pipeline. `FileEntry` includes rich metadata: `purpose` (e.g., "header file", "C++20 module interface", "test file") and `declarations` (e.g., `["class basic_json", "function parse"]`).

### 6.4 Intent Types

| Intent | Trigger | Handler |
|---|---|---|
| `QUESTION` | Default; asking about code | `answer_question` |
| `COMPILE` | Build/compile keywords | `run_build` |
| `RUN` | Test/run/execute keywords | `run_tests` |
| `EXPLORE` | List/search/find/browse keywords | `explore_codebase` |
| `EXIT` | exit/quit/bye/done | Graph terminates |

---

## 7. Sub-Agent Architecture

Each sub-agent is an independent LLM call with its own 5000-token budget. They expand the system's effective processing capacity without violating the per-call token constraint.

### 7.1 Retrieval Compressor

**Purpose:** Read ~3000 tokens of raw code and produce a ~400-token dense digest.

**System Prompt Rules:**
1. Focus on what's RELEVANT to the user's question
2. Preserve key facts: file paths, function/class names, line numbers, signatures, return types
3. Use compact notation, skip boilerplate
4. Output under `max_output_tokens`
5. Say "INSUFFICIENT: <reason>" if code doesn't answer the question
6. Never answer the question — just summarize relevant code

**Fallback:** On failure, return trimmed raw code.

### 7.2 Tool Output Summarizer

**Purpose:** Condense large shell command outputs (build logs, test results) into ~200-token summaries.

**Trigger:** `should_summarize_tool_output(content)` — checks for `[stdout]`/`[stderr]` markers and length > 800 chars.

**System Prompt Rules:**
1. Preserve ALL critical info: error messages, file paths, pass/fail counts, exit codes
2. Remove repetitive output (compiler flags, progress bars)
3. Build output: focus on errors, warnings, success/failure status
4. Test output: total/passed/failed counts, failing test names
5. Structured format: status line first, then key details

**Mechanism:** Creates replacement `ToolMessage` with compressed content and same `tool_call_id`/`id` — the `add_messages` reducer merges by ID, effectively replacing the original.

### 7.3 Conversation Compressor

**Purpose:** Re-summarize `summary_of_knowledge` every 3 turns to keep it tight and relevant.

**Trigger:** `turn_count > 1 and turn_count % 3 == 0` (in `classify_and_prepare`)

**Input:** Previous summary + last 6 messages (3 human+AI turn pairs, content capped at 300 chars each)

**System Prompt Rules:**
1. Preserve ALL discovered facts: file purposes, architecture decisions, build results, test outcomes
2. Track what user asked and what was answered
3. Note current build state
4. Drop conversational fluff
5. Use bullet points grouped by topic

### 7.4 Multi-Hop Decomposer

**Purpose:** Handle complex multi-aspect questions by decomposition → parallel investigation → merge.

**Flow:**
1. **Decompose** (1 LLM call): Break complex question into 2-3 independent sub-queries (JSON array output)
2. **Investigate** (N parallel LLM calls): Each sub-query runs a retrieval subagent. If `index_search_fn` is provided, each sub-query gets its own targeted code chunks from the index.
3. **Merge** (1 LLM call): Merge sub-findings into one coherent digest, eliminating duplicates

**Total calls:** 2 + N (decompose + N investigations + merge). Typically 4-5 calls.

**Trigger:** `is_complex_question(user_query)` — heuristic based on question marks, comparison keywords, multiple file references, or long multi-clause queries.

**Fallback:** If all sub-findings fail, falls back to a single retrieval subagent call.

---

## 8. Tool Reference

### 8.1 `execute_shell_command`

```python
@tool
def execute_shell_command(cmd: str) -> str:
```

Run a shell command via `subprocess.run(shell=True)`. Output format:

```
command=<normalized_cmd>
result=PASS|FAIL (exit_code=N)
[tests=X% passed, Y failed out of Z]  # if CTest detected
[error_hint=<first error line>]        # if non-zero exit
[cmd]=<cmd>
[exit_code]=N
[stdout]
...
[stderr]
...
```

- On Windows: command normalized via `_normalize_command_for_platform()`
- Output smart-truncated to 3000 chars: head 40 lines + critical lines + tail 40 lines

### 8.2 `list_directory`

```python
@tool
def list_directory(path: str, depth: int = 1) -> str:
```

List directory contents recursively up to `depth` (max 3). Directories sorted first, then files. Indented tree format.

### 8.3 `read_file_chunk`

```python
@tool
def read_file_chunk(filepath: str, start_line: int, end_line: int) -> str:
```

Read a file chunk by 1-indexed line range (max 250 lines). Returns prefixed lines: `N: content`.

### 8.4 `search_codebase`

```python
@tool
def search_codebase(regex_pattern: str) -> str:
```

Search source files with regex (case-insensitive). Returns grep-like `path:line:content` matches. Max 60 matches. Uses `_iter_text_files()` respecting `SKIP_DIRS`/`SKIP_EXTENSIONS`. Lines truncated at 200 chars. Uses `get_workspace_root()` for base directory.

---

## 9. Build State Machine

The `BuildState` tracks the project's build lifecycle across turns.

```
┌──────┐   cmake OK    ┌─────────────┐   make/ninja OK   ┌──────────┐   ctest OK   ┌─────────┐
│ IDLE │──────────────→│ CONFIGURING  │──────────────────→│ BUILDING │────────────→│ SUCCESS │
└──────┘               └─────────────┘                   └──────────┘             └─────────┘
   │                         │                                │                        │
   │     any cmd fails       │      any cmd fails             │    ctest fails         │
   └─────────┬───────────────┴────────────────────────────────┴───────────────────────→│
             ↓                                                                         │
         ┌────────┐                                                                    │
         │ FAILED │ ←─────────────────────────────────────────────────────────────────┘
         └────────┘   (consecutive_errors++, last_error set)
```

- `consecutive_errors` resets to 0 on any successful command
- `last_exit_code` tracks the most recent command's exit code
- `configured`, `built`, `tested` booleans accumulate (monotonic unless reset)

---

## 10. Configuration Reference

All configuration is via `AgentConfig` (frozen dataclass). Values are populated from environment variables with sensible defaults.

| Environment Variable | Default | Type | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | (required) | `str` | OpenAI API key |
| `AGENT_MODEL` | `gpt-5.3-codex` | `str` | Main reasoning model |
| `AGENT_CLASSIFIER_MODEL` | `gpt-4o-mini` | `str` | Intent classifier (lightweight, ~110 tokens) |
| `AGENT_SUBAGENT_MODEL` | `gpt-5.3-codex` | `str` | Sub-agent model |
| `AGENT_TOKEN_BUDGET` | `5000` | `int` | Total token budget (must be 5000) |
| `AGENT_INPUT_TOKENS` | `4000` | `int` | Input token budget |
| `AGENT_OUTPUT_TOKENS` | `1000` | `int` | Output token budget (capped to 800 in practice) |
| `AGENT_MAX_TOOL_ITERATIONS` | `3` | `int` | Max ReAct tool loop iterations per turn |
| `AGENT_WORKSPACE_PATH` | `workspace/json` | `Path` | Path to nlohmann/json codebase |
| `AGENT_USE_RETRIEVAL_SUBAGENT` | `true` | `bool` | Enable retrieval compression sub-agent |
| `AGENT_USE_TOOL_SUMMARIZER` | `true` | `bool` | Enable tool output summarizer sub-agent |
| `AGENT_USE_CONVERSATION_COMPRESSOR` | `true` | `bool` | Enable conversation compressor sub-agent |
| `AGENT_USE_MULTI_HOP` | `true` | `bool` | Enable multi-hop decomposition for complex questions |
| `AGENT_RETRIEVAL_DIGEST_TOKENS` | `400` | `int` | Max tokens for retrieval digest output |
| `AGENT_TOOL_SUMMARY_TOKENS` | `200` | `int` | Max tokens for tool summary output |

---

## 11. Test Suite

Located in `tests/`. Run with:

```bash
pytest tests/ -v
```

| Test File | Coverage |
|---|---|
| `test_config.py` | `AgentConfig` default construction, frozen immutability, budget invariant validation, `token_budget == 5000` enforcement, `effective_output_budget` cap |
| `test_state.py` | `BuildState` default values, frozen immutability (cannot set `status`, `built`), construction with values, equality/inequality |
| `test_token_utils.py` | `estimate_text_tokens` (empty/short/long), `trim_text_to_token_budget` (empty budget, within budget, long text trimming, prefix preservation, binary search tightness), `estimate_token_count` (empty/single/multiple/tool_calls), `fit_messages_to_budget` (within budget, drops oldest, drops tool pairs first, preserves minimum), `sanitize_tool_message_sequence` (orphan removal, matched kept, non-tool preserved) |
| `test_tools.py` | `_truncate_output` (short/long/critical patterns/hard truncation), `_normalize_command_for_platform` (Windows: `./`, `export`, `rm -rf`), `_extract_ctest_summary` (valid/failed/none), `_extract_error_hint` (error/no error), `set_workspace_root`/`get_workspace_root` (set+get, fallback to cwd) |
| `test_intent.py` | `_fallback_classify` (exit/build/run/explore/question keywords), `_fallback_followup` (confirm/cancel/exit/new_request, empty string) |
| `test_model_utils.py` | `is_responses_model` (chat models return False, responses models return True, edge cases), `extract_text` (string/list-of-dicts/list-of-strings/mixed/empty/other), `normalize_ai_message` (string/list content, tool_calls preservation) |

---

## 12. Coding Conventions

### Python Style
- Python 3.10+ with `from __future__ import annotations`
- Type hints on all function signatures (use `|` union syntax)
- `@dataclass(frozen=True)` for immutable config/state objects
- `TypedDict` for LangGraph state
- f-strings for string formatting
- Imports at module level

### LangGraph Patterns
- Nodes are plain functions: `def node_name(state: AgentState, config: AgentConfig) -> dict[str, Any]`
- Nodes return partial state dicts (only keys they update)
- `add_messages` reducer for the messages list
- Conditional edges use router functions returning string node names
- Tools use `@tool` decorator from `langchain_core.tools`

### Token Management
- Always use `estimate_token_count()` to measure message lists
- Always use `trim_text_to_token_budget()` to cap text
- Always call `fit_messages_to_budget()` before invoking the LLM
- Reserve ~300 tokens for tool schemas when tools are bound

### Tool Design
- Structured output with `[cmd]=`, `[exit_code]=`, `[stdout]`, `[stderr]` markers
- Smart-truncated: head 40 lines + critical lines + tail 40 lines
- Windows command normalization via `_normalize_command_for_platform()`
- Max 3000 chars after truncation

### Error Handling
- LLM failures produce fallback `AIMessage` with error details
- Build/test failures tracked via `BuildState.consecutive_errors`
- ReAct loop enforces `max_tool_iterations`
- Sub-agent failures fall back to trimmed raw content

### Model Utilities
- ALL `ChatOpenAI` instances via `build_chat_model()`
- `is_responses_model()` auto-detects API mode with compiled regex patterns
- `normalize_ai_message()` handles list-of-blocks → str
- `run_async()` handles event loop edge cases

---

## 13. Dependencies

```
langgraph>=0.2.53        — State graph, tool nodes, message reducers
langchain-core>=0.3.0    — Base message types, @tool decorator
langchain-openai>=0.2.0  — ChatOpenAI with Responses API support
tiktoken>=0.8.0          — Token counting (cl100k_base encoder)
python-dotenv>=1.0.1     — .env file loading
```

---

## 14. Important Rules & Invariants

1. **Never exceed the 5000-token budget.** Every main LLM call is measured and fit within budget programmatically. Sub-agents are separate calls with their own 5000-token caps.
2. **Never clone or download repositories.** The agent operates exclusively on the pre-downloaded local copy at `workspace/json/`.
3. **Never modify target repository source code.** The agent reads, builds, and tests — it does not edit the user's project files.
4. **Commands must actually execute.** All build/test commands run locally via `subprocess` — never simulated.
5. **Always ground answers in evidence.** If the agent doesn't have enough context, it uses tools or says so.
6. **Intent classification uses a separate LLM call.** Lightweight classifier (~110 tokens) is outside the main budget.
7. **The conversation loop never auto-terminates.** Only the user's "exit" intent ends the session.
8. **Context retrieval is per-turn.** Each turn fetches fresh relevant snippets.
9. **Conversation history uses sliding window + summary.** Last 2-3 full turns; older compressed into `summary_of_knowledge`.
10. **Windows compatibility is required.** All shell commands normalized. Use `os.name == "nt"` checks.
11. **`workspace/json/` is the target codebase directory.** Verified at startup.
12. **Config is immutable after construction.** Use `dataclasses.replace()` for modified copies.

---

## Project Structure

```
main.py                     # REPL entry point
agent/
  __init__.py               # Package exports: build_init_graph, build_turn_graph, AgentConfig
  config.py                 # AgentConfig — frozen dataclass, env var overrides, budget validation
  state.py                  # AgentState (TypedDict), BuildState, FileEntry, SymbolEntry, CodebaseIndex
  graph.py                  # LangGraph StateGraph definitions (init + per-turn)
  nodes.py                  # Graph nodes, routers, LLM invocation, environment probing, build state tracking
  intent.py                 # Intent classifier (async LLM + keyword fallback) + follow-up classifier
  indexer.py                # Codebase indexer (file manifest, symbols, purpose map, search, fuzzy matching)
  tools.py                  # LangChain tools: execute_shell_command, list_directory, read_file_chunk, search_codebase
  prompts.py                # Intent-specific system prompts (QUESTION, COMPILE, RUN, EXPLORE)
  subagents.py              # Sub-agents: retrieval compressor, tool summarizer, conversation compressor, multi-hop
  model_utils.py            # Model detection, ChatOpenAI construction, response normalization, async helpers
  token_utils.py            # Token counting (tiktoken), binary-search trimming, budget fitting, message sanitization
tests/
  __init__.py
  test_config.py            # AgentConfig validation and immutability tests
  test_state.py             # BuildState immutability tests
  test_token_utils.py       # Token counting, trimming, budget fitting, sanitization tests
  test_tools.py             # Output truncation, command normalization, CTest parsing, workspace root tests
  test_intent.py            # Keyword fallback classification tests
  test_model_utils.py       # Model detection, text extraction, message normalization tests
requirements.txt            # Python dependencies
README.md                   # Project overview and usage guide
document.md                 # This file — comprehensive technical reference
workspace/json/             # Pre-downloaded nlohmann/json codebase (not agent source)
logs/                       # Runtime logs directory
```
