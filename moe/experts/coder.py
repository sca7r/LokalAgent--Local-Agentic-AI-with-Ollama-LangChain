"""
moe/experts/coder.py

WHY generic: builds Python variable block from context respecting value_type.
Numbers become floats, text becomes strings, lists become Python lists.
No assumptions about what previous steps returned.
"""
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from moe.memory_manager import switch_to, unload_current
from moe.config_loader import get_coder_model, get_preferred_language
from moe.expert_contract import ok, err


def run(instruction: str, context: dict = None) -> dict:
    lang  = get_preferred_language()
    model = get_coder_model()
    ctx   = context or {}

    switch_to(model)
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)

    # Build variable block respecting value_type
    # WHY: number → float literal, text → string, list → Python list
    var_lines = []
    for k in sorted(ctx):
        if not k.endswith("_value"):
            continue
        val   = ctx[k]
        vtype = ctx.get(k.replace("_value", "_value_type"), "")
        if val is None:
            continue
        if vtype == "number" or (vtype == "" and _is_numeric(val)):
            var_lines.append(f"{k} = {float(val)}")
        elif vtype == "list" and isinstance(val, list):
            var_lines.append(f"{k} = {val!r}")
        elif vtype == "dict" and isinstance(val, dict):
            var_lines.append(f"{k} = {val!r}")
        else:
            var_lines.append(f'{k} = "{val}"')

    var_block = "\n".join(var_lines)

    prompt = f"""Write {lang} code to: {instruction}

{"Available variables from previous steps:" if var_block else ""}
{var_block}

Rules:
- Use the variables above directly, do NOT redefine them
- End with a clear print() of the result
- Write ONLY executable code, no markdown fences, no explanation

Code:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.strip()
    if "```" in code:
        code = "\n".join(l for l in code.split("\n") if not l.strip().startswith("```"))
    code = code.strip()
    if var_block:
        code = f"{var_block}\n{code}"

    print(f"[Coder] Language: {lang}")
    print(f"[Coder] Code:\n{code[:300]}")

    try:
        from langchain_experimental.tools import PythonREPLTool
        output = PythonREPLTool().run(code).strip()
        unload_current()

        # Return with correct value_type
        nums = re.findall(r'-?\d+(?:\.\d+)?', output)
        if nums and output.replace(nums[0],'').strip() in ('', '.'):
            return ok(value=float(nums[0]), value_type="number",
                      summary=f"Code result: {output[:100]}",
                      expert="code", task=instruction, raw=output)
        return ok(value=output, value_type="text",
                  summary=f"Code result: {output[:100]}",
                  expert="code", task=instruction, raw=output)

    except Exception as e:
        unload_current()
        return err(str(e), expert="code", task=instruction, raw=code)


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False
