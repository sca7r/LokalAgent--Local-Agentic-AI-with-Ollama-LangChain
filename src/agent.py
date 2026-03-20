"""
Agentic AI — LangChain ReAct Agent with 4 tools (Ollama backend)
=================================================================
Tools:  web_search | python_repl | read_file | write_file | api_call
LLM:    Ollama (local) — no API key needed
Memory: sliding-window conversation buffer

Prerequisites:
    1. Ollama installed and running:  https://ollama.com
    2. Model pulled:  ollama pull llama3.2
    3. pip install -r requirements.txt

Usage:
    python agent.py
"""

import sys
import os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain import hub

from config import OLLAMA_BASE_URL, MODEL_NAME, MAX_ITERATIONS, VERBOSE
from memory import get_memory
from tools.search import get_search_tool
from tools.code_exec import get_code_exec_tool
from tools.file_ops import get_file_tools
from tools.api_call import get_api_tool


def build_agent() -> AgentExecutor:
    """Assemble and return a ready-to-run ReAct agent."""

    # 1. LLM — local Ollama, no API key required
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,        # deterministic reasoning
    )

    # 2. Tools
    search = get_search_tool()
    tools = [
        *([search] if search else []),
        get_code_exec_tool(),
        *get_file_tools(),
        get_api_tool(),
    ]

    # 3. Prompt  — pull the standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react-chat")

    # 4. Agent logic
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 5. Executor (adds memory, iteration cap, error handling)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=get_memory(),
        verbose=VERBOSE,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,   # recover from malformed LLM output
        return_intermediate_steps=False,
    )


def main():
    print(f"\n  Agentic AI — LangChain ReAct Agent  [{MODEL_NAME} via Ollama]")
    print("Tools: web_search | python_repl | read_file | write_file | api_call")
    print("Type 'exit' or 'quit' to stop.\n")

    agent_executor = build_agent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {result['output']}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()