# Context-Constrained AI Codebase Assistant

A human-interactive AI coding assistant built with **LangGraph** that helps users understand, build, and test the **nlohmann/json** C++ library while operating under a strict **5000-token LLM context limit**.

## What This Project Does

- Works with a **pre-downloaded** local copy of [nlohmann/json](https://github.com/nlohmann/json) at `workspace/json/`
- Runs as a **conversational REPL** — ask questions, request builds, run tests interactively
- Classifies user intent (question, build, test, explore) via a lightweight parallel LLM call
- Retrieves relevant code context using a **three-layer pipeline**: path-aware direct load → keyword/symbol search → ReAct tool fallback
- Compresses code and tool outputs via **sub-agent LLM calls** to fit within the token budget
- Executes actual build/test commands locally (`cmake`, `ninja`, `ctest`) and interprets results
- Maintains a rolling conversation summary across turns — never loses context

## Architecture at a Glance

```
Startup:  START → index_workspace → END

Per-turn:
  START → classify_and_prepare → retrieve_context → route_by_intent:
    ├── answer_question  ──┐
    ├── run_build        ──┤── route_after_llm ──→ execute_tools → handle_tool_result
    ├── run_tests        ──┤                         → continue_or_respond (loop)
    ├── explore_codebase ──┘
    └── exit → END

Sub-agents (separate LLM calls, outside main budget):
    ├── Retrieval Compressor  (code → dense digest)
    ├── Tool Output Summarizer (build logs → compact summary)
    ├── Conversation Compressor (rolling summary refresh)
    └── Multi-Hop Decomposer  (complex questions → parallel investigate → merge)
```

## Key Implementation Details

- **Hard 5000-token budget**: enforced programmatically — includes system prompt, context, history, tool schemas, and output
- Default budgets: input 4000, output 1000 (capped to 800 in practice)
- All intents use a **ReAct tool loop** (max 3 iterations) — the LLM can read files, list directories, search code, and run shell commands
- Sub-agents expand effective capacity to ~20,000+ tokens per turn across multiple independent LLM calls
- Token counting via `tiktoken`; pruning drops oldest tool-call/observation pairs first
- Smart output truncation preserves critical lines (errors, test summaries, CTest counts)
- Windows-compatible: command normalization, environment probing, path handling

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` environment variable set
- Build tooling: `cmake` and at least one backend (`ninja`, `make`, or `mingw32-make`)

## Quick Start (From Scratch)

### 1) Clone and enter the project

```bash
git clone https://github.com/<your-username>/limited_token_agent_code_runner.git
cd limited_token_agent_code_runner
```

### 2) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Set required environment variables

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

macOS/Linux:

```bash
export OPENAI_API_KEY=sk-...
```

Optional (if your default model is unavailable):

Windows PowerShell:

```powershell
$env:AGENT_MODEL="gpt-4o-mini"
```

macOS/Linux:

```bash
export AGENT_MODEL=gpt-4o-mini
```

### 5) Verify required local codebase exists

This project expects a pre-downloaded target repo at `workspace/json/`.

```bash
# from project root
ls workspace/json
```

If `workspace/json/` is missing, place your local nlohmann/json copy there before running the agent.

### 6) Run the agent

```bash
python main.py
```

### 7) Try a smoke test in the REPL

- `What does include/nlohmann/json.hpp do?`
- `build the project`
- `run ctest`
- `exit`

## Configuration

Set your API key and optionally configure via `.env` or environment variables:

```bash
# Unix/macOS
export OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

Key environment variables (see `agent/config.py` for all options):

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `gpt-5.3-codex` | Main reasoning model |
| `AGENT_CLASSIFIER_MODEL` | `gpt-4o-mini` | Intent classifier model |
| `AGENT_SUBAGENT_MODEL` | `gpt-5.3-codex` | Sub-agent model |
| `AGENT_WORKSPACE_PATH` | `workspace/json` | Path to nlohmann/json codebase |
| `AGENT_MAX_TOOL_ITERATIONS` | `3` | Max ReAct tool loop iterations per turn |
| `AGENT_USE_RETRIEVAL_SUBAGENT` | `true` | Enable retrieval compression sub-agent |
| `AGENT_USE_MULTI_HOP` | `true` | Enable multi-hop decomposition |

Note: the default `AGENT_MODEL` is tuned for this assignment setup; if unavailable in your account, override it in `.env` (for example: `AGENT_MODEL=gpt-4o-mini`).

## Usage

```bash
python main.py
```

Interactive commands:
- Ask questions about code: `"What does json.hpp do?"`, `"Explain the parser architecture"`
- Build: `"build the project"`, `"compile with ninja"`
- Test: `"run the tests"`, `"run ctest"`
- Explore: `"list the src directory"`, `"search for parse functions"`
- Exit: `"exit"`, `"quit"`, `"bye"`

## Project Structure

```
main.py                 # REPL entry point — input loop, graph invocation, output display
agent/
  __init__.py           # Package exports
  config.py             # AgentConfig — immutable settings with env var overrides
  state.py              # AgentState, BuildState, FileEntry, SymbolEntry, CodebaseIndex
  graph.py              # LangGraph StateGraph definitions (init + per-turn)
  nodes.py              # Graph node functions, routers, LLM invocation, helpers
  intent.py             # Intent classifier (async LLM + keyword fallback) + follow-up classifier
  indexer.py            # Codebase indexer (file manifest, symbols, purpose map, search)
  tools.py              # LangChain tools: shell, read_file_chunk, list_directory, search_codebase
  prompts.py            # Intent-specific system prompts
  subagents.py          # Sub-agent modules (retrieval, tool summarizer, compressor, multi-hop)
  model_utils.py        # Model detection, ChatOpenAI construction, response normalization
  token_utils.py        # Token counting, trimming, budget fitting, message sanitization
document.md             # Comprehensive technical reference (architecture, data flow, module docs)
report.md               # Technical report version for supervisor review
README.md
requirements.txt
workspace/json/         # Pre-downloaded nlohmann/json codebase (not agent source)
logs/
```

## Design Document

See [document.md](document.md) for the complete architecture walkthrough, including:
- Module-by-module reference
- Data flow diagrams
- Token budget deep dive
- Three-layer retrieval pipeline
- Sub-agent architecture
- Build state machine
- Configuration reference

## Notes

- The agent operates **exclusively** on the pre-downloaded `workspace/json/` — it never clones or downloads repositories.
- Commands are executed locally via `subprocess` — never simulated.
- The agent reads, builds, and tests — it **never modifies** the target project's source code.
- The 5000-token limit is enforced programmatically per LLM call. Sub-agents are separate calls with their own 5000-token caps.
