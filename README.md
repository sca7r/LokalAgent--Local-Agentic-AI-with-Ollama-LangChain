# LokalAgent 
### Local Agentic AI with Ollama & LangChain

LokalAgent is a fully local, privacy-first agentic AI system. It runs open-source LLMs entirely on your machine via Ollama — no API keys, no cloud dependency — with a Claude-style chat UI, multi-agent collaboration, and PageIndex-style vectorless RAG for PDF analysis.

---

##  Features

- **100% local LLM** via Ollama (llama3, phi3, mistral, qwen2.5, and more)
- **Single-agent mode** — ReAct reasoning loop with 5 built-in tools
- **Multi-agent mode** — Planner → Executor → Reviewer pipeline with self-correction
- **RAG** — vectorless, reasoning-based PDF search (no embeddings, no vector DB)
- **PDF upload in UI** — attach PDFs directly in the chat interface
- **Web search** — real-time internet access via Tavily
- **Zero cloud cost** — runs entirely offline (except optional web search)

---

## Architecture

### Single Agent (ReAct)
```
User → ReAct Agent → [Think → Tool → Observe] loop → Final Answer
```

### Multi-Agent Pipeline
```
User
 ↓
Planner     — breaks task into ordered steps with tool assignments
 ↓
Executor    — runs each step, extracts clean variables, chains results
 ↓
Reviewer    — evaluates results, passes or sends feedback to Planner
 ↓ (loop up to 3 iterations)
Final Answer
```

### RAG
```
PDF Upload
 ↓
Indexer     — tags pages as <physical_index_X>, detects TOC (Table of Contents) (3 paths),
              builds hierarchical tree, verifies + self-corrects indices
 ↓
JSON Tree Index (cached)
 ↓
Retriever   — iterative loop: read TOC → select section → fetch text →
              check if sufficient → repeat → generate answer
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain (ReAct) |
| LLM backend | Ollama |
| Multi-agent | Custom Planner / Executor / Reviewer |
| RAG | Custom style (vectorless) |
| Web search | Tavily API |
| Code execution | LangChain Python REPL |
| Web UI | Flask + Vanilla JS |
| Language | Python 3.10+ |

---

## Project Structure

```
LokalAgent/
├── src/
│   ├── app.py              # Flask backend - all API routes
│   ├── agent.py            # CLI agent entrypoint
│   └── index.html          # Claude-style chat UI
├── agents/
│   ├── planner.py          # Planner agent - task → JSON plan
│   ├── executor.py         # Executor agent - runs tools, chains results
│   ├── reviewer.py         # Reviewer agent - evaluates, gives feedback
│   └── orchestrator.py     # Coordinates the 3-agent pipeline
├── rag/
│   ├── indexer.py          # PDF tree builder
│   ├── retriever.py        # Iterative tree-search retriever
│   └── utils.py            # LLM calls, PDF parsing, token counting
├── tools/
│   ├── search.py           # Web search (Tavily)
│   ├── code_exec.py        # Python REPL
│   ├── file_ops.py         # File read/write
│   ├── api_call.py         # HTTP API calls
│   └── doc_search.py       # PDF document search tool
├── config.py               # Model name, Ollama URL, settings
├── memory.py               # Sliding-window conversation memory
├── requirements.txt
└── .env.example
```

---

##  Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Clone the repository

```bash
git clone https://github.com/your-username/LokalAgent.git
cd LokalAgent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull a model via Ollama

```bash
ollama pull llama3       # recommended for best results
# or
ollama pull phi3:mini    # lighter, faster on low-RAM machines
```

### 5. (Optional) Enable web search

Sign up free at [tavily.com](https://tavily.com) and add your key:

```bash
cp .env.example .env
# Edit .env and add: TAVILY_API_KEY=tvly-your-key-here
```

### 6. Run

```bash
python3 src/app.py
```

Open **http://localhost:5000** in your browser.

---

##  Usage

### Single agent mode
Ask anything directly:
```
What is the weather in Bengaluru right now?
Calculate compound interest on ₹50,000 at 8% for 5 years
Fetch the GitHub profile for torvalds via API
```

### Multi-agent mode
Switch to **Multi agent** in the header and give multi-step tasks:
```
Search for the latest Bitcoin price, calculate 15% of it, and save to crypto.txt
Find the top 3 Python web frameworks and write a comparison to frameworks.md
```

### PDF upload
Click the 📎 paperclip → select a PDF → wait for indexing → ask questions:
```
What are the key findings in this document?
Summarise section 3
What does the document say about methodology?
```

---

##  Configuration

Edit `config.py`:

```python
MODEL_NAME       = "llama3:latest"   # any Ollama model
OLLAMA_BASE_URL  = "http://localhost:11434"
MAX_ITERATIONS   = 10                # single agent max steps
```

---

##  Known Limitations

### Multi-agent with small models (phi3:mini)
- phi3:mini often fails to generate valid JSON plans — falls back to keyword-based planning
- Calculation steps may be inaccurate when the LLM misreads extracted values
- **Workaround:** Use `llama3:latest` for multi-agent tasks

### RAG indexing speed
- Indexing is slow on CPU-only machines (no GPU)
- A 10-page PDF takes 3–5 minutes on an i7 with 8GB RAM
- **Workaround:** Indexed PDFs are cached — re-uploading the same PDF loads instantly

### LLM hallucination in multi-agent pipelines
- When chaining steps, the LLM occasionally generates incorrect Python code
- The Reviewer catches most errors and re-runs the pipeline (up to 3 iterations)
- Complex calculations are more reliable with llama3 than phi3

### Context window
- Ollama's default context window is 4096 tokens
- Very long search results or large PDFs may be truncated
- **Workaround:** The RAG indexer chunks documents into sections to stay within limits

---

##  TODO

### High priority
- [ ] **Docker deployment** - single `docker-compose up` to run everything
- [ ] **Fix multi-agent calculation accuracy** - replace LLM code generation with a sandboxed Python evaluator that parses the task directly
- [ ] **Streaming responses** - stream LLM tokens to the UI instead of waiting for the full response
- [ ] **Better context chaining** - structured JSON passing between agent steps instead of string injection

### Features
- [ ] **Conversation history persistence** - save and reload past conversations
- [ ] **Tool usage analytics dashboard** - track which tools are used, success rates, latency
- [ ] **Custom tool builder** - add new tools from the UI without editing code
- [ ] **Voice input** - speak your queries using browser speech API
- [ ] **Multiple PDF support** - index and search across multiple documents simultaneously

### RAG improvements
- [ ] **Async indexing** - show real-time progress while PDF is being indexed
- [ ] **Vision RAG** - support scanned PDFs using page image analysis
- [ ] **Cross-document references** - follow "see Appendix G" style links between sections

### Model & performance
- [ ] **GPU acceleration** - auto-detect and use GPU via Ollama
- [ ] **Model benchmarking** - compare phi3 vs llama3 vs mistral on standard tasks
- [ ] **Prompt caching** - cache repeated LLM calls to speed up re-runs

---

##  License

MIT License — free to use, modify, and distribute.

---

##  Acknowledgements

- [LangChain](https://langchain.com) — agent framework
- [Ollama](https://ollama.com) — local LLM runtime
- [Tavily](https://tavily.com) — AI-optimised web search
- [PageIndex by VectifyAI](https://github.com/VectifyAI/PageIndex) - inspiration for vectorless RAG architecture
