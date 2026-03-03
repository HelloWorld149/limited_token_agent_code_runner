# Context-Constrained LangGraph Build Agent

ReAct-style build/test agent that clones a target repository, explores it, runs configure/build/test commands, and produces a final evidence-based report while operating under a strict 5000-token budget.

## What this project does

- Clones a repository (default: `https://github.com/nlohmann/json`) into `workspace/<repo-name>`
- Detects local environment/tooling before reasoning (OS, cmake, g++, MSVC, ninja, make, mingw32-make)
- Uses LangGraph looped reasoning with tool calls to discover, build, test, and diagnose failures
- Prunes chat history when over budget and preserves key findings as summary breadcrumbs
- Generates a final report from retained command evidence (with local fallback if LLM report generation fails)

## Current architecture at a glance

```
START
  |
  v
initialize_workspace
  |
  v
agent_reasoner --(tool calls)--> execute_tools --> context_manager --+
      |                                                     |
      +------------------(no tool / step limit)------------+
                                |
                                v
                          generate_report --> END
```

ReAct cycle: `agent_reasoner -> execute_tools -> context_manager -> agent_reasoner`.

## Key implementation details

- Hard budget invariant enforced in config: `token_budget == 5000` and `input + output <= total`
- Default budgets: input 4000, output 1000, prune threshold 4000
- Reasoner output budget is capped to <=700 tokens; report can use up to output budget
- Failure handling is reflective (single reconsideration pass) rather than forced command injection
- Context pruning protects important tool outputs (test summaries, failures, recent tool-call context)
- Command output truncation keeps head/tail plus critical lines (CTest/CMake/error signals)

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` configured
- `git`
- Build tooling appropriate for your target project (`cmake` and at least one build backend)

## Installation

```bash
git clone https://github.com/<your-username>/limited_token_agent_code_runner.git
cd limited_token_agent_code_runner
pip install -r requirements.txt
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="<your-key>"
```

## Usage

Basic run:

```bash
python main.py
```

Verbose stream with logging:

```bash
python main.py --verbose-loop --log-file logs/run.log
```

Common options:

```bash
python main.py \
  --model gpt-5.3-codex \
  --max-steps 25 \
  --repo-dir workspace \
  --clone-url https://github.com/nlohmann/json \
  --verbose-loop \
  --log-file logs/run.log
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-5.3-codex` | LLM model name |
| `--max-steps` | `25` | Maximum reasoning loops |
| `--repo-dir` | `workspace` | Parent directory for cloning |
| `--clone-url` | `https://github.com/nlohmann/json` | Target repository URL |
| `--verbose-loop` | `false` | Stream per-node updates |
| `--log-file` | `""` | Optional log file path |

Environment variable overrides are supported via `AGENT_*` settings in `agent/config.py`.

## Project structure

```
main.py
agent/
  config.py
  graph.py
  nodes.py
  prompts.py
  state.py
  token_utils.py
  tools.py
ARCHITECTURE.md
README.md
requirements.txt
workspace/            # runtime clone location (contains target repo, e.g. workspace/json)
logs/
```

## Notes

- This agent is designed to inspect/build/test target repositories and report findings; it should not edit target source code.
- `workspace/json` is a runtime clone destination for the default target and not required as committed source for the Python agent itself.
- For exact node-by-node behavior, see `ARCHITECTURE.md`.
