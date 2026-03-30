"""
moe/experts/file_expert.py

WHY generic: converts any value_type to appropriate file content.
number → "result: 471.66 USD/oz"
text   → the text directly
list   → one item per line
dict   → key: value pairs
"""
import os, re
from moe.expert_contract import ok, err


def read(instruction: str, context: dict = None) -> dict:
    path = instruction.strip().strip('"\'')
    if not os.path.exists(path):
        return err(f"File not found: {path}", expert="file_read", task=instruction)
    try:
        with open(path) as f:
            content = f.read()
        return ok(value=content, value_type="text",
                  summary=f"Read {len(content)} chars from '{path}'",
                  expert="file_read", task=instruction, raw=content)
    except Exception as e:
        return err(str(e), expert="file_read", task=instruction)


def write(instruction: str, context: dict = None) -> dict:
    ctx = context or {}

    # Substitute {stepN_value} placeholders
    for k, v in ctx.items():
        instruction = instruction.replace(f"{{{k}}}", str(v))

    # Extract filename
    fname_match = re.search(r'[\w\-]+\.\w+', instruction)
    fname = fname_match.group(0) if fname_match else "output.txt"

    # Build content from explicit "|||" separator or from context
    if "|||" in instruction:
        _, content = instruction.split("|||", 1)
    else:
        content = _build_content_from_context(ctx)

    if not content.strip():
        return err("No content to write", expert="file_write", task=instruction)

    return _save(fname, content, instruction)


def _build_content_from_context(ctx: dict) -> str:
    """
    Convert context values to human-readable file content.
    WHY generic: handles number, text, list, dict value_types.
    """
    lines = []
    steps = sorted(set(
        k.split("_")[0] for k in ctx if k.startswith("step")
    ))
    for step in steps:
        val   = ctx.get(f"{step}_value")
        vtype = ctx.get(f"{step}_value_type", "")
        unit  = ctx.get(f"{step}_unit", "")
        summ  = ctx.get(f"{step}_summary", "")

        if val is None:
            continue

        if vtype == "number":
            lines.append(f"{summ or step}: {val} {unit}".strip())
        elif vtype == "list" and isinstance(val, list):
            lines.append(f"{summ or step}:")
            for item in val:
                lines.append(f"  - {item}")
        elif vtype == "dict" and isinstance(val, dict):
            lines.append(f"{summ or step}:")
            for k, v in val.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{summ or step}: {val}")

    return "\n".join(lines)


def _save(path: str, content: str, task: str = "") -> dict:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return ok(value=path, value_type="file",
                  summary=f"Saved {len(content)} chars to '{path}'",
                  expert="file_write", task=task, raw=content)
    except Exception as e:
        return err(str(e), expert="file_write", task=task)
