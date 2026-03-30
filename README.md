# LokalAgent 🤖

**A fully local agentic AI system with Mixture of Experts (MoE) routing, structured JSON inter-agent communication, and constrained LLM outputs, running entirely on your machine via Ollama.**

No cloud. No API costs. No data leaves your device.

---

## What is this?

LokalAgent is a local AI assistant built around two modes:

- **Normal mode** - deepseek-r1 answers directly with streaming, just like ChatGPT
- **Think mode** - activates a full MoE pipeline where a router breaks your task into steps, specialist experts handle each step, and a reviewer validates the result

The key technical idea: instead of one general-purpose model doing everything poorly, LokalAgent routes each sub-task to the most capable specialist, a coding model for code, a pure Python engine for math, Tavily for live search and passes structured JSON between them so nothing gets lost in translation.

---

## Architecture

```
User Query
    │
    ├── Normal Mode ──→ deepseek-r1:7b (streaming, conversational)
    │
    └── Think Mode ───→ Router (deepseek-r1)
                            │
                            ▼
                    ┌───────────────────────────────────┐
                    │  Specialist Experts               │
                    │  ┌─────────┐  ┌───────────────┐  │
                    │  │web_srch │  │  math (Python)│  │
                    │  │(Tavily) │  │  no LLM       │  │
                    │  └─────────┘  └───────────────┘  │
                    │  ┌─────────┐  ┌───────────────┐  │
                    │  │  code   │  │  file_write   │  │
                    │  │(qwen2.5)│  │  pure Python  │  │
                    │  └─────────┘  └───────────────┘  │
                    │  ┌─────────┐  ┌───────────────┐  │
                    │  │doc_srch │  │  api_call     │  │
                    │  │(RAG PDF)│  │  HTTP client  │  │
                    │  └─────────┘  └───────────────┘  │
                    └───────────────────────────────────┘
                            │
                            ▼
                    Reviewer (deepseek-r1)
                            │
                     pass? ─┴─ fail → retry with fix instructions (max 3)
                            │
                    Humaniser (deepseek-r1)
                            │
                    Clean answer to user
```

### RAM Management

Only **one LLM lives in RAM at a time**. Ollama's `keep_alive=0` unloads each model immediately after use. deepseek-r1 (router/reviewer) and qwen2.5-coder (code steps) swap in and out as needed, making the system work on machines with 8–10GB RAM.

---

## Key Technical Decisions

### Structured JSON Inter-Expert Communication

Every expert sends and receives typed JSON matching a contract schema:

```json
{
  "status":     "ok",
  "expert":     "web_search",
  "task":       "find current gold price",
  "result": {
    "value":      4623.93,
    "value_type": "number",
    "unit":       "USD/oz",
    "summary":    "Gold spot price is $4,623.93 per troy ounce as of March 22 2026"
  },
  "confidence": "high",
  "error":      "",
  "raw":        "..."
}
```

This eliminates regex-based value extraction between steps, the math expert reads `step1_value = 4623.93` (a Python float), not `"Gold spot price is $4,623.93..."` (a string to parse).

### Constrained LLM Outputs via Ollama Structured Outputs

The router and reviewer use Ollama's `format` parameter with Pydantic schemas. This passes the schema to llama.cpp as a GBNF grammar, invalid tokens are masked at sampling time. The model cannot produce malformed JSON.

Nested Pydantic models generate `$defs`/`$ref` in JSON Schema, which Ollama's grammar engine doesn't support. `schema_utils.py` resolves all references inline before passing to Ollama, preserving full schema structure without breaking constrained decoding.

### Search → Extract → Use Pipeline

Web search follows a clean three-step pipeline:

1. **Fetch** — Tavily retrieves raw text from the web (no LLM)
2. **Extract** — deepseek reads raw text + task context → `ExtractionResult` JSON with typed value
3. **Pass** — clean typed value (`4623.93`, not a sentence) flows to the next expert

This means math, file, and code experts always receive clean typed data, they never parse raw search text.

### PageIndex RAG (Vectorless PDF Search)

PDF search uses a reasoning-based approach inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex). Pages are tagged with `<physical_index_X>` markers, a hierarchical TOC is built via LLM, and retrieval iterates: read TOC → select section → fetch text → check sufficient → repeat. No embeddings, no vector database.

---

## Project Structure

```
LokalAgent/
├── src/
│   ├── app.py              # Flask backend - /chat (stream) and /think (MoE+SSE)
│   └── index.html          # Chat UI
│
├── moe/
│   ├── router.py           # deepseek-r1: task → ordered plan (constrained JSON)
│   ├── reviewer.py         # deepseek-r1: evaluate results, generate fix instructions
│   ├── orchestrator.py     # coordinates router → experts → reviewer loop
│   ├── expert_contract.py  # JSON schema, sanitization, call_with_schema()
│   ├── schema_utils.py     # $defs/$ref resolver for Ollama compatibility
│   ├── memory_manager.py   # Ollama model load/unload (one model in RAM at a time)
│   ├── config_loader.py    # reads moe_config.yaml
│   ├── moe_config.yaml     # models, experts, behaviour - edit without touching code
│   └── experts/
│       ├── searcher.py     # Tavily fetch + LLM extract → typed JSON
│       ├── math_engine.py  # pure Python arithmetic - no LLM, no hallucination
│       ├── coder.py        # qwen2.5-coder:7b - generates + runs code
│       ├── file_expert.py  # pure Python read/write
│       ├── api_expert.py   # HTTP calls
│       └── doc_expert.py   # PageIndex RAG for PDFs
│
├── rag/
│   ├── indexer.py          # PageIndex-style PDF tree builder
│   ├── retriever.py        # iterative tree-search retrieval
│   └── utils.py            # LLM helpers, page tagging, token counting
│
├── tools/                  # LangChain tool wrappers (used by single agent)
├── moe_config.yaml         # model + expert configuration
└── requirements.txt
```

---

## Models

| Role | Model | Why |
|---|---|---|
| Router + Reviewer + Direct answers | `deepseek-r1:7b` | Chain-of-thought reasoning, self-verification |
| Code generation | `qwen2.5-coder:7b` | Trained specifically on code, fewer syntax errors |
| Math | Python `sympy` + stdlib | No hallucination possible, pure computation |
| Web search | Tavily API | Real-time data, no LLM needed |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone

```bash
git clone https://github.com/your-username/LokalAgent.git
cd LokalAgent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull models

```bash
ollama pull deepseek-r1:7b
ollama pull qwen2.5-coder:7b
```

### 4. Configure web search (optional)

Sign up free at [tavily.com](https://tavily.com), then:

```bash
cp .env.example .env
# Add your key: TAVILY_API_KEY=tvly-your-key-here
```

### 5. Run

```bash
python3 src/app.py
```

Open **http://localhost:5000**

---

## Usage

### Normal mode
Type any question, deepseek-r1 answers with streaming, showing its thinking process in a collapsible panel.

```
What is quantum entanglement?
Write a poem about the ocean
Explain the difference between TCP and UDP
```

### Think mode
Click the **✨ Think** button to activate the MoE pipeline for multi-step tasks:

```
Search the current gold price, calculate 10%, and save to gold.txt
Find the top 5 Python web frameworks and write a comparison to frameworks.md
What is the weather in Mumbai today?
```

### PDF upload
Click 📎 → select a PDF → wait for indexing → ask questions about it:

```
Summarise this document
What does section 3 say about methodology?
List all the key findings
```

---

## Configuration

Edit `moe/moe_config.yaml` to change models or behaviour, no code changes needed:

```yaml
models:
  router:   "deepseek-r1:7b"
  coder:    "qwen2.5-coder:7b"
  fallback: "llama3:latest"

user:
  preferred_language: "python"   # code generation language

behaviour:
  max_retries: 3
  unload_after_use: true
```

---

## Honest Limitations

| Limitation | Detail |
|---|---|
| **Model quality** | deepseek-r1:7b is a 7B parameter model. GPT-4/Claude use 100B+. Answers are less reliable on complex reasoning. |
| **RAM** | Running two 4.7GB models requires 8GB+ RAM. Only one loads at a time. |
| **Speed** | Each LLM call takes 8–20s on CPU. Multi-step tasks can take 2–5 minutes. |
| **Constrained outputs** | Schema is enforced at token level for router/reviewer. Other experts use prompt-based JSON with sanitization fallback. |
| **Web search** | Requires Tavily API key and internet access. |
| **RAG indexing** | Slow on CPU, a 10-page PDF takes 3–5 minutes to index. Cached after first run. |

---

## What I Learned Building This

- **MoE at agent level** — routing to specialist models/tools is more reliable than asking one model to do everything
- **Structured outputs matter** — the difference between asking an LLM to "return JSON" vs enforcing the schema at token level is significant
- **$defs in JSON Schema** — Pydantic's nested model schemas use `$ref` pointers that Ollama's grammar engine can't parse; they need to be resolved inline before passing to the constrained decoder
- **Separation of concerns** — searcher fetches and extracts, math computes, file writes; each expert does one thing well
- **Small models are opinionated** — deepseek-r1:7b follows instructions reliably but phi3:mini does not; model selection is as important as architecture

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM backend | Ollama |
| Constrained outputs | Ollama structured outputs + llama.cpp GBNF grammars |
| Schema validation | Pydantic v2 |
| Agent framework | LangChain (streaming, tools) |
| Web search | Tavily API |
| RAG | Custom PageIndex-style (no vector DB) |
| Web UI | Flask + Vanilla JS (SSE streaming) |
| Math | Python stdlib + sympy |

---

## Acknowledgements

- [Ollama](https://ollama.com) — local LLM runtime
- [deepseek-r1](https://github.com/deepseek-ai/DeepSeek-R1) — reasoning model
- [qwen2.5-coder](https://github.com/QwenLM/Qwen2.5-Coder) — coding model  
- [Tavily](https://tavily.com) — AI-optimised web search API
- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — inspiration for vectorless RAG
- [LangChain](https://langchain.com) — agent and tool framework

---

## License

MIT — free to use, modify, and distribute.