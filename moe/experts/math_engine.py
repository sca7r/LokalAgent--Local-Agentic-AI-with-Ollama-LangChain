"""
moe/experts/math_engine.py

WHY generic: checks value_type from context before operating.
If step1 is "text" (a list of frameworks), math skips it.
If step1 is "number" (a price), math uses it directly.
No hardcoded assumptions about what the previous step returned.
"""
import re
from moe.expert_contract import ok, err


def run(instruction: str, context: dict = None) -> dict:
    ctx  = context or {}
    inst = instruction.lower()

    try:
        # Find the base numeric value from context
        # WHY: only use context values that are actually numbers
        base_val  = None
        base_unit = ""
        for step_key in sorted(k for k in ctx if k.endswith("_value")):
            vtype = ctx.get(step_key.replace("_value", "_value_type"), "")
            val   = ctx[step_key]
            if vtype == "number" or (vtype == "" and _is_numeric(val)):
                base_val  = float(val)
                unit_key  = step_key.replace("_value", "_unit")
                base_unit = ctx.get(unit_key, "")
                break

        # If no numeric context, extract from instruction text
        if base_val is None:
            nums = re.findall(r'-?\d+(?:\.\d+)?', instruction)
            if nums:
                base_val = float(nums[0])

        if base_val is None:
            return err("No numeric value available to calculate with",
                       expert="math", task=instruction)

        # Detect operation
        pct = re.search(r'(\d+(?:\.\d+)?)\s*%', inst)
        if pct:
            p      = float(pct.group(1))
            result = base_val * (p / 100)
            return ok(value=round(result, 6), value_type="number",
                      unit=base_unit,
                      summary=f"{p}% of {base_val} {base_unit} = {round(result, 6)} {base_unit}",
                      expert="math", task=instruction,
                      raw=str(result))

        nums = re.findall(r'-?\d+(?:\.\d+)?', instruction)

        if any(w in inst for w in ['add', 'plus', 'sum', '+']):
            b = float(nums[1]) if len(nums) >= 2 else 0
            r = base_val + b
            return ok(value=round(r,6), value_type="number", unit=base_unit,
                      summary=f"{base_val} + {b} = {r}", expert="math", task=instruction, raw=str(r))

        if any(w in inst for w in ['subtract', 'minus', 'difference']):
            b = float(nums[1]) if len(nums) >= 2 else 0
            r = base_val - b
            return ok(value=round(r,6), value_type="number", unit=base_unit,
                      summary=f"{base_val} - {b} = {r}", expert="math", task=instruction, raw=str(r))

        if any(w in inst for w in ['multiply', 'times', '* ']):
            b = float(nums[1]) if len(nums) >= 2 else 1
            r = base_val * b
            return ok(value=round(r,6), value_type="number", unit=base_unit,
                      summary=f"{base_val} × {b} = {r}", expert="math", task=instruction, raw=str(r))

        if any(w in inst for w in ['divide', 'divided', 'ratio']):
            b = float(nums[1]) if len(nums) >= 2 else 1
            if b == 0:
                return err("Division by zero", expert="math", task=instruction)
            r = base_val / b
            return ok(value=round(r,6), value_type="number", unit=base_unit,
                      summary=f"{base_val} ÷ {b} = {r:.6f}", expert="math", task=instruction, raw=str(r))

        import math as _math
        if 'sqrt' in inst or 'square root' in inst:
            r = _math.sqrt(base_val)
            return ok(value=round(r,6), value_type="number", unit=base_unit,
                      summary=f"√{base_val} = {r:.6f}", expert="math", task=instruction, raw=str(r))

        return err(f"Cannot determine operation from: {instruction}",
                   expert="math", task=instruction)

    except Exception as e:
        return err(str(e), expert="math", task=instruction)


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False
