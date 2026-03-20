"""
LokalAgent GUI — Flask backend
"""

import sys, os

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from flask import Flask, request, jsonify, send_from_directory
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain import hub

from tools.code_exec import get_code_exec_tool
from tools.file_ops import get_file_tools
from tools.api_call import get_api_tool
from tools.search import get_search_tool
from tools.doc_search import get_doc_search_tool, set_index, clear_index
from agents.orchestrator import run_multi_agent

import threading, json

app = Flask(__name__)

MODELS       = ["phi3:mini", "llama3:latest"]
UPLOAD_DIR   = os.path.join(_root, "uploads")
INDEX_DIR    = os.path.join(_root, "indexes")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)

agents:   dict = {}
memories: dict = {}


# ── Callback to capture tool usage ──────────────────────────────────────────
class ToolCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.tools_used = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tools_used.append(serialized.get("name", "unknown"))


# ── Agent builder ────────────────────────────────────────────────────────────
def build_agent(model_name: str) -> AgentExecutor:
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0,
    )
    tools = [
        get_doc_search_tool(),
        get_code_exec_tool(),
        *get_file_tools(),
        get_api_tool(),
    ]
    search = get_search_tool()
    if search:
        tools.insert(0, search)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=10, return_messages=True
    )
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(
        "You are a concise AI assistant. Answer the user's question directly and briefly.\n\n"
        "You have access to these tools:\n{tools}\n\n"
        "Use this format:\n"
        "Question: the input question\n"
        "Thought: think about what to do\n"
        "Action: tool name, one of [{tool_names}]\n"
        "Action Input: input to the tool\n"
        "Observation: result of the tool\n"
        "Thought: I now have enough information to answer\n"
        "Final Answer: give a SHORT, direct answer using the observation\n\n"
        "IMPORTANT: After getting an Observation, go straight to Final Answer. Do not write articles or long explanations.\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )
    agent  = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    memories[model_name] = memory
    return AgentExecutor(
        agent=agent, tools=tools, memory=memory,
        verbose=True, max_iterations=10,
        handle_parsing_errors=True,
    )


def get_agent(model_name: str) -> AgentExecutor:
    if model_name not in agents:
        agents[model_name] = build_agent(model_name)
    return agents[model_name]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/models")
def list_models():
    return jsonify({"models": MODELS})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Receive a PDF, build PageIndex tree, register doc_search tool."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save PDF
    pdf_path   = os.path.join(UPLOAD_DIR, file.filename)
    index_path = os.path.join(INDEX_DIR,  file.filename.replace(".pdf", ".json"))
    file.save(pdf_path)

    model = request.form.get("model", MODELS[0])

    # Load cached index if it exists
    if os.path.exists(index_path):
        with open(index_path) as f:
            doc_index = json.load(f)
        set_index(doc_index)
        return jsonify({
            "status":   "indexed",
            "filename": file.filename,
            "pages":    doc_index["total_pages"],
            "nodes":    doc_index["total_nodes"],
            "cached":   True,
        })

    # Build new index
    try:
        from rag.indexer import build_index
        doc_index = build_index(
            pdf_path=pdf_path,
            model_name=model,
            index_path=index_path,
        )
        set_index(doc_index)
        return jsonify({
            "status":   "indexed",
            "filename": file.filename,
            "pages":    doc_index["total_pages"],
            "nodes":    doc_index["total_nodes"],
            "cached":   False,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.json
    user_msg = data.get("message", "").strip()
    model    = data.get("model", MODELS[0])

    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    cb = ToolCaptureCallback()
    try:
        executor = get_agent(model)
        result   = executor.invoke(
            {"input": user_msg},
            config={"callbacks": [cb]}
        )
        return jsonify({
            "response":   result.get("output", ""),
            "tools_used": cb.tools_used,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_memory():
    data  = request.json or {}
    model = data.get("model", MODELS[0])
    if model in memories:
        memories[model].clear()
    if model in agents:
        del agents[model]
    clear_index()
    return jsonify({"status": "cleared"})


@app.route("/multi-agent", methods=["POST"])
def multi_agent():
    data     = request.json
    task     = data.get("message", "").strip()
    model    = data.get("model", MODELS[0])

    if not task:
        return jsonify({"error": "empty task"}), 400

    try:
        result = run_multi_agent(task=task, model=model)
        # Format a rich response showing the plan + final answer
        plan_summary = "\n".join([
            f"Step {s.get('step')}: {s.get('description')} [{s.get('tool')}]"
            for s in result["plan"]
        ])
        response_text = (
            f"**Plan ({result['iterations']} iteration(s)):**\n{plan_summary}\n\n"
            f"**Result:**\n{result['final_answer']}"
        )
        return jsonify({
            "response":   response_text,
            "tools_used": result["tools_used"],
            "iterations": result["iterations"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("LokalAgent GUI → http://localhost:5000")
    app.run(debug=True, port=5000, threaded=True, request_handler=None)

# Increase werkzeug timeout for slow LLM responses
import socket
socket.setdefaulttimeout(600)  # 10 minutes
