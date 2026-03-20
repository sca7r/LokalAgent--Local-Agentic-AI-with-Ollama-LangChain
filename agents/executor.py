"""
agents/executor.py

Key insight: after each step, ask the LLM ONE focused question:
"What is the key value/result from this output?" → store as clean variable.
Next step gets: previous_variables + task description → write code.

This separates extraction (easy) from generation (hard) into two small prompts
instead of one big hallucination-prone prompt.
"""

import sys, os
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def extract_key_result(output: str, description: str, model: str) -> str:
    """
    Ask the LLM ONE focused question: what is the key value from this output?
    Returns a short, clean answer — a number, a word, a date.
    This is a much easier task than writing code, so hallucination is minimal.
    """
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)
    prompt = f"""From this text, extract ONLY the key numerical value or result relevant to: "{description}"

Text: {output[:600]}

Reply with ONLY the value itself. Examples:
- "5602.22" (for a price)
- "March 20 2026" (for a date)
- "560.22" (for a calculated result)

No explanation. Just the value:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().split('\n')[0].strip()


def build_tool_map() -> dict:
    from tools.search     import get_search_tool
    from tools.code_exec  import get_code_exec_tool
    from tools.file_ops   import get_file_tools
    from tools.api_call   import get_api_tool
    from tools.doc_search import get_doc_search_tool

    tool_map = {}
    for t in get_file_tools():
        tool_map[t.name] = t.run
    tool_map["python_repl"]     = get_code_exec_tool().run
    tool_map["api_call"]        = get_api_tool().run
    tool_map["document_search"] = get_doc_search_tool().func

    search = get_search_tool()
    if search:
        tool_map["web_search"] = search.run
    return tool_map


def build_step_input(step: dict, extracted_vars: dict, model: str) -> str:
    """
    Build the tool input for this step.
    - python_repl / write_file: give the LLM clean extracted variables + task
    - web_search / api_call: use original input as-is
    """
    tool        = step.get("tool", "")
    description = step.get("description", "")
    original    = step.get("input", "")

    if not extracted_vars:
        return original

    if tool == "python_repl":
        import re

        def clean_value(v: str) -> str:
            """Convert extracted value to a valid Python literal."""
            stripped = v.strip().replace(",", "")
            # Remove currency symbols and units
            stripped = re.sub(r"^[\$\£\€\¥]", "", stripped)
            stripped = re.sub(r"[a-zA-Z%°]+$", "", stripped).strip()
            try:
                float(stripped)
                return stripped   
            except ValueError:
                return f'"{v.strip()}"'  

        # Build variable assignments from extracted results
        var_block = "\n".join(
            f"{name} = {clean_value(value)}"
            for name, value in extracted_vars.items()
        )
        llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)
        prompt = f"""Write Python code to: {description}

These variables are already available (use them directly, do NOT redefine them):
{var_block}

Rules:
- Use the variables above directly
- End with a print() statement showing the result
- Write ONLY the code, no explanation, no markdown fences

Code:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content.strip()
        # Strip markdown fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(l for l in lines if not l.startswith("```"))
        return f"{var_block}\n{code}"

    if tool == "write_file":
        import re
        fname_match = re.search(r'[\w\-]+\.\w+', original + " " + description)
        fname = fname_match.group(0) if fname_match else "output.txt"
        # Build clean readable content
        content_lines = []
        for k, v in extracted_vars.items():
            content_lines.append(f"{k}: {v}")
        # Also add a summary line
        vals = list(extracted_vars.values())
        if len(vals) >= 2:
            content_lines.append(f"\nSummary:")
            content_lines.append(f"  Input value : {vals[0]}")
            content_lines.append(f"  Result      : {vals[-1]}")
        return f"{fname}|||" + "\n".join(content_lines)

    return original


def run_executor(plan: list, model: str = "phi3:mini") -> list:
    """
    Execute each step.
    After each step, extract the key result into a named variable.
    Pass those clean variables to subsequent steps.
    """
    tool_map      = build_tool_map()
    results       = []
    extracted_vars = {}   # name → clean value, passed to subsequent steps

    for step in plan:
        step_num    = step.get("step", "?")
        tool_name   = step.get("tool", "none")
        description = step.get("description", "")

        print(f"[Executor] Step {step_num}: {description} (tool: {tool_name})")

        if tool_name == "none" or not tool_name:
            results.append({"step": step_num, "tool": "none",
                            "input": step.get("input",""),
                            "output": step.get("input",""), "success": True})
            continue

        if tool_name not in tool_map:
            results.append({"step": step_num, "tool": tool_name,
                            "input": step.get("input",""),
                            "output": f"Tool '{tool_name}' not available.", "success": False})
            continue

        # Build input using clean extracted vars from previous steps
        tool_input = build_step_input(step, extracted_vars, model)
        print(f"[Executor] Input: {tool_input[:120].replace(chr(10),' | ')}...")

        try:
            output = tool_map[tool_name](tool_input)
            output_str = str(output)[:3000]
            results.append({"step": step_num, "tool": tool_name,
                            "input": tool_input, "output": output_str, "success": True})
            print(f"[Executor] Step {step_num} ✓  →  {output_str[:80]}")

            # Extract key result and store as named variable for next steps
            if tool_name in ("web_search", "python_repl", "api_call", "document_search"):
                var_name = f"step{step_num}_result"
                key_val  = extract_key_result(output_str, description, model)
                extracted_vars[var_name] = key_val
                print(f"[Executor] Extracted: {var_name} = {key_val}")

        except Exception as e:
            results.append({"step": step_num, "tool": tool_name,
                            "input": tool_input, "output": f"Error: {e}", "success": False})
            print(f"[Executor] Step {step_num} ✗ — {e}")

    return results
