"""
LokalAgent — Flask backend
Two modes:
  Normal : deepseek-r1 answers directly (fast, conversational)
  Think  : MoE pipeline (router → experts → reviewer)
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from moe.orchestrator import run_moe
from moe.config_loader import load as load_config
from tools.doc_search import set_index, clear_index

import threading, json, re

app  = Flask(__name__)

DIRECT_MODEL   = "deepseek-r1:7b"
SYSTEM_PROMPT  = (
    "You are LokalAgent, a helpful local AI assistant built with Ollama. "
    "Answer concisely and directly. "
    "If you don't know something or need real-time data, say so clearly."
)

# Conversation memory per session
_sessions: dict = {}   # session_id → list of messages


def get_history(session_id: str) -> list:
    return _sessions.get(session_id, [])


def add_to_history(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
    # Keep last 20 messages
    _sessions[session_id] = _sessions[session_id][-20:]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Normal mode — deepseek-r1 answers directly with streaming."""
    data       = request.json
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    add_to_history(session_id, "user", user_msg)
    history = get_history(session_id)

    def generate():
        llm = ChatOllama(
            model=DIRECT_MODEL,
            base_url="http://localhost:11434",
            temperature=0.7,
        )

        # Build message list
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history[:-1]:   # exclude current message (already added)
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_msg))

        full_response = ""
        thinking_text = ""
        in_think = False

        try:
            print(f"\n[deepseek] Thinking: ", end='', flush=True)
            for chunk in llm.stream(messages):
                token = chunk.content
                if not token:
                    continue

                # Handle <think> blocks from deepseek-r1
                if "<think>" in token:
                    in_think = True
                if "</think>" in token:
                    in_think = False
                    # Send end of thinking signal
                    yield f"data: {json.dumps({'type': 'think_end'})}\n\n"
                    continue

                if in_think:
                    thinking_text += token
                    yield f"data: {json.dumps({'type': 'think', 'token': token})}\n\n"
                else:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            add_to_history(session_id, "assistant", full_response)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/think", methods=["POST"])
def think():
    """Think mode — full MoE pipeline with SSE progress streaming."""
    data     = request.json
    task     = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not task:
        return jsonify({"error": "empty task"}), 400

    def generate():
        import queue
        q = queue.Queue()

        def callback(event):
            q.put(event)

        def worker():
            try:
                result = run_moe(task=task, progress_callback=callback)
                q.put({
                    "type":       "done",
                    "response":   result["final_answer"],
                    "tools_used": result["tools_used"],
                    "attempts":   result["attempts"],
                })
                add_to_history(session_id, "user", task)
                add_to_history(session_id, "assistant", result["final_answer"])
            except Exception as e:
                q.put({"type": "error", "message": str(e)})

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event = q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400

    UPLOAD_DIR = os.path.join(_root, "uploads")
    INDEX_DIR  = os.path.join(_root, "indexes")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR,  exist_ok=True)

    pdf_path   = os.path.join(UPLOAD_DIR, file.filename)
    index_path = os.path.join(INDEX_DIR,  file.filename.replace(".pdf", ".json"))
    file.save(pdf_path)

    if os.path.exists(index_path):
        with open(index_path) as f:
            doc_index = json.load(f)
        set_index(doc_index)
        return jsonify({"status": "indexed", "filename": file.filename,
                        "pages": doc_index["total_pages"],
                        "nodes": doc_index["total_nodes"], "cached": True})
    try:
        from rag.indexer import build_index
        doc_index = build_index(pdf_path=pdf_path, model_name=DIRECT_MODEL, index_path=index_path)
        set_index(doc_index)
        return jsonify({"status": "indexed", "filename": file.filename,
                        "pages": doc_index["total_pages"],
                        "nodes": doc_index["total_nodes"], "cached": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    data       = request.json or {}
    session_id = data.get("session_id", "default")
    if session_id in _sessions:
        del _sessions[session_id]
    clear_index()
    return jsonify({"status": "cleared"})


@app.route("/set-language", methods=["POST"])
def set_language():
    lang = request.json.get("language", "python")
    import moe.config_loader as cl
    import yaml
    cfg = cl.load()
    if "user" not in cfg:
        cfg["user"] = {}
    cfg["user"]["preferred_language"] = lang
    config_path = os.path.join(_root, "moe", "moe_config.yaml")
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        if "user" not in raw:
            raw["user"] = {}
        raw["user"]["preferred_language"] = lang
        with open(config_path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"[Config] Could not persist language: {e}")
    return jsonify({"status": "ok", "language": lang})


if __name__ == "__main__":
    print(f"LokalAgent → http://localhost:5000  |  Model: {DIRECT_MODEL}")
    app.run(debug=True, port=5000, threaded=True)
