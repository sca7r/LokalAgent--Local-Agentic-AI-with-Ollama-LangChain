"""
moe/expert_contract.py

Production-grade constrained JSON contract using Ollama structured outputs.

HOW IT WORKS (from research):
- Ollama passes the Pydantic schema to llama.cpp as a GBNF grammar
- During token sampling, any token that would violate the schema is MASKED
- The model cannot produce invalid JSON — it's enforced at the token level
- This is the same mechanism OpenAI and Anthropic use internally

WHY Pydantic:
- model_json_schema() generates the exact format Ollama expects
- model_validate_json() parses AND validates the response
- If validation fails, we know immediately and can retry
"""

from __future__ import annotations
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# ── Pydantic models (the contract) ───────────────────────────────────────────

class ExpertResultData(BaseModel):
    """The actual result payload from an expert."""
    value: Any = Field(
        description="The actual result. Number for prices/counts, string for facts, list for multiple items, dict for structured data, null if failed"
    )
    value_type: Literal["number", "text", "list", "dict", "file", "bool", "null"] = Field(
        description="Python type of value: number=float/int, text=string, list=array, dict=object, file=filepath, null=no result"
    )
    unit: str = Field(
        default="",
        description="Unit of measurement e.g. 'USD/oz', 'km/h', '%' — empty string if no unit applies"
    )
    summary: str = Field(
        description="One clear sentence describing the result e.g. 'Gold spot price is $4716.56 per troy ounce as of Mar 22 2026'"
    )


class ExpertMessage(BaseModel):
    """
    Standard inter-expert message. Every expert produces this.
    Enforced at token level via Ollama structured outputs.
    """
    status: Literal["ok", "error"] = Field(
        description="Whether the expert completed its task successfully"
    )
    expert: str = Field(
        description="Name of the expert e.g. web_search, math, file_write, code"
    )
    task: str = Field(
        description="The exact instruction this expert received"
    )
    result: ExpertResultData = Field(
        description="The result payload"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How reliable: high=verified data, medium=likely correct, low=uncertain"
    )
    error: str = Field(
        default="",
        description="Error message if status is error, empty string if ok"
    )
    raw: str = Field(
        default="",
        description="Full raw output truncated to 500 chars for debugging"
    )


# ── Helper to call Ollama with constrained output ────────────────────────────

def _resolve_refs(schema: dict) -> dict:
    """
    Inline all $ref references in a JSON Schema.

    WHY: Ollama's llama.cpp grammar engine cannot handle $ref pointers.
    Pydantic generates $defs + $refs for nested models (e.g. list[RouterStep]).
    We resolve all references inline so Ollama gets a fully self-contained
    schema — semantically identical to the original, just without pointers.

    This is how you keep the full nested JSON structure (like GPT/Claude use)
    while still being compatible with Ollama's constrained decoding.
    """
    defs = schema.get("$defs", {})

    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve(defs[ref_name])
                return obj
            return {k: resolve(v) for k, v in obj.items() if k != "$defs"}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)


def _sanitize_pydantic(obj: BaseModel) -> BaseModel:
    """
    Sanitize all constrained fields on a Pydantic model instance.
    Modifies in place and returns the object.
    Works generically on any model — no schema-specific code needed.
    """
    for field_name in obj.model_fields:
        raw = getattr(obj, field_name, None)
        if raw is not None and isinstance(raw, str):
            sanitized = sanitize_field(field_name, raw)
            if sanitized != raw:
                print(f"[Contract] Sanitized {field_name}: '{raw}' → '{sanitized}'")
                object.__setattr__(obj, field_name, sanitized)
    return obj


def call_with_schema(model: str, messages: list, schema_model: type[BaseModel]) -> BaseModel | None:
    """
    Call Ollama with constrained JSON output.

    HOW:
    1. Generate full nested JSON Schema from Pydantic model
    2. Resolve all $ref/$defs inline (Ollama can't handle pointers)
    3. Pass resolved schema to Ollama format parameter
    4. llama.cpp converts to GBNF grammar — invalid tokens MASKED at generation
    5. Validate response with Pydantic — guaranteed correct structure

    WHY resolve_refs not flatten:
    Flattening loses the semantic structure (nested objects become flat fields).
    Resolving keeps the full structure intact — same schema GPT/Claude use,
    just with references inlined rather than referenced.
    """
    import re, json

    schema = _resolve_refs(schema_model.model_json_schema())
    print(f"[Contract] Schema resolved — no $defs: {'$defs' not in json.dumps(schema)}")

    # ── Try native ollama package (constrained decoding) ─────────────────
    try:
        import ollama as _ollama
        for attempt in range(2):
            try:
                response = _ollama.chat(
                    model=model,
                    messages=messages,
                    format=schema,
                    options={"temperature": 0},
                )
                text = response.message.content
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                parsed = schema_model.model_validate_json(text)
                # Sanitize constrained fields at contract level — works for all schemas
                return _sanitize_pydantic(parsed)
            except Exception as e:
                print(f"[Contract] Ollama attempt {attempt+1} failed: {e}")
    except ImportError:
        print("[Contract] ollama package not installed — using LangChain fallback")
        print("[Contract] Run: pip install ollama pydantic")

    # ── Fallback: LangChain + prompt-based JSON ───────────────────────────
    # Not token-constrained but works without ollama package
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)
        schema_str = json.dumps(schema, indent=2)  # already dereffed

        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=
                    m["content"] + f"\n\nReturn ONLY valid JSON matching this schema:\n{schema_str}"
                ))

        response  = llm.invoke(lc_messages)
        text = response.content.strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Extract JSON block
        start = text.find('{'); end = text.rfind('}')
        if start != -1 and end != -1:
            parsed = schema_model.model_validate_json(text[start:end+1])
            return _sanitize_pydantic(parsed)
    except Exception as e:
        print(f"[Contract] LangChain fallback failed: {e}")

    return None


# ── Schema sanitisation ──────────────────────────────────────────────────────

# For each field that has a known valid set, define:
# - valid_values: the allowed set
# - fallback: what to use if none match
# - keywords: partial match hints (e.g. "fail" in "task failed" → "fail")
_FIELD_CONSTRAINTS = {
    "confidence": {
        "valid":    {"high", "medium", "low"},
        "fallback": "medium",
        "keywords": {"high": ["high", "very reliable", "authoritative", "official"],
                     "low":  ["low", "uncertain", "unclear", "estimated"],
                     "medium": ["medium", "moderate", "general"]},
    },
    "verdict": {
        "valid":    {"pass", "fail"},
        "fallback": "fail",
        "keywords": {"pass": ["pass", "success", "correct", "complete", "ok"],
                     "fail": ["fail", "error", "wrong", "missing", "incomplete"]},
    },
    "task_type": {
        "valid":    {"simple", "multi_step"},
        "fallback": "multi_step",
        "keywords": {"simple":     ["simple", "single", "one step", "direct"],
                     "multi_step": ["multi", "step", "multiple", "complex"]},
    },
    "value_type": {
        "valid":    {"number", "text", "list", "dict", "file", "bool", "null"},
        "fallback": "text",
        "keywords": {"number": ["number", "numeric", "float", "int", "price", "count"],
                     "list":   ["list", "array", "multiple", "items"],
                     "dict":   ["dict", "object", "map", "key"],
                     "file":   ["file", "path", "filepath"],
                     "bool":   ["bool", "boolean", "true", "false"],
                     "null":   ["null", "none", "empty", "no result"]},
    },
    "status": {
        "valid":    {"ok", "error"},
        "fallback": "error",
        "keywords": {"ok":    ["ok", "success", "done", "complete"],
                     "error": ["error", "fail", "exception", "invalid"]},
    },
}


def sanitize_field(field_name: str, raw_value: str) -> str:
    """
    Normalize a field value to its valid set using exact match, then keyword match.

    WHY generic:
    Without Literal type constraints, the LLM can write anything.
    Instead of patching each field individually (which breaks for new examples),
    we define constraints once here and apply them universally.

    Works for ANY field in ANY expert result — not just gold price.
    e.g. sanitize_field("confidence", ":star:") → "medium"
    e.g. sanitize_field("task_type", "steps") → "multi_step"
    e.g. sanitize_field("verdict", "task completed") → "pass"
    """
    if field_name not in _FIELD_CONSTRAINTS:
        return raw_value  # no constraint defined — return as-is

    constraint = _FIELD_CONSTRAINTS[field_name]
    val = str(raw_value).lower().strip()

    # 1. Exact match
    if val in constraint["valid"]:
        return val

    # 2. Keyword match — check if any valid value's keywords appear in the raw value
    for valid_val, keywords in constraint["keywords"].items():
        if any(kw in val for kw in keywords):
            return valid_val

    # 3. Fallback
    return constraint["fallback"]


def sanitize_result(result: dict) -> dict:
    """
    Sanitize all known constrained fields in an expert result.
    Called automatically after every call_with_schema().

    WHY at contract level not expert level:
    If we sanitize in each expert, we'd need to patch every file every time.
    Centralizing here means ALL experts get correct values automatically.
    New fields can be added to _FIELD_CONSTRAINTS without touching any expert.
    """
    for field in ["confidence", "status", "value_type"]:
        if field in result:
            result[field] = sanitize_field(field, result[field])
        # Also check nested result dict
        if "result" in result and isinstance(result["result"], dict):
            if field in result["result"]:
                result["result"][field] = sanitize_field(field, result["result"][field])
    return result


# ── Builder functions (backward compat + flat field helpers) ─────────────────

def _infer_type(value: Any) -> str:
    if value is None:                   return "null"
    if isinstance(value, bool):         return "bool"
    if isinstance(value, (int, float)): return "number"
    if isinstance(value, list):         return "list"
    if isinstance(value, dict):         return "dict"
    return "text"


def ok(value: Any, unit: str = "", summary: str = "",
       expert: str = "", task: str = "", raw: str = "",
       value_type: str = None, confidence: str = "high") -> dict:
    vtype = value_type or _infer_type(value)
    summ  = summary or str(value)[:120]
    return {
        "status": "ok", "expert": expert, "task": task,
        "result": {"value": value, "value_type": vtype, "unit": unit, "summary": summ},
        "confidence": confidence, "error": "", "raw": (raw or str(value))[:500],
        "success": True, "output": summ,
        "value": value, "value_type": vtype, "unit": unit, "summary": summ,
    }


def err(message: str, expert: str = "", task: str = "", raw: str = "") -> dict:
    return {
        "status": "error", "expert": expert, "task": task,
        "result": {"value": None, "value_type": "null", "unit": "", "summary": f"Failed: {message}"},
        "confidence": "low", "error": message, "raw": (raw or message)[:500],
        "success": False, "output": f"Failed: {message}",
        "value": None, "value_type": "null", "unit": "", "summary": f"Failed: {message}",
    }


def get_context_vars(results: list) -> dict:
    ctx = {}
    for r in results:
        step  = r.get("step", "?")
        res   = r.get("result") or {}
        val   = r.get("value")      if "value"      in r else res.get("value")
        vtype = r.get("value_type") if "value_type" in r else res.get("value_type", _infer_type(val))
        unit  = r.get("unit",  "")  if "unit"   in r else res.get("unit",  "")
        summ  = r.get("summary","") if "summary" in r else res.get("summary","")
        conf  = r.get("confidence","medium")

        if r.get("status") == "ok" or r.get("success"):
            if val is not None:
                ctx[f"step{step}_value"]      = val
                ctx[f"step{step}_value_type"] = vtype
                ctx[f"step{step}_unit"]       = unit
                ctx[f"step{step}_summary"]    = summ
                ctx[f"step{step}_confidence"] = conf
    return ctx


def validate(result: dict) -> tuple[bool, str]:
    required = ["status", "expert", "task", "result", "confidence", "error", "raw"]
    for f in required:
        if f not in result:
            return False, f"Missing field: {f}"
    if result["status"] not in ("ok", "error"):
        return False, f"Invalid status: {result['status']}"
    if result["confidence"] not in ("high", "medium", "low"):
        return False, f"Invalid confidence: {result['confidence']}"
    res = result.get("result", {})
    for f in ["value", "value_type", "unit", "summary"]:
        if f not in res:
            return False, f"Missing result.{f}"
    valid_types = {"number", "text", "list", "dict", "file", "bool", "null"}
    if res.get("value_type") not in valid_types:
        return False, f"Invalid value_type: {res.get('value_type')}"
    return True, ""


def format_for_terminal(results: list) -> str:
    lines = []
    for r in results:
        step   = r.get("step", "?")
        expert = r.get("expert", "?")
        status = r.get("status", "ok" if r.get("success") else "error")
        res    = r.get("result") or {}
        val    = r.get("value") if "value" in r else res.get("value")
        vtype  = r.get("value_type","") if "value_type" in r else res.get("value_type","")
        unit   = r.get("unit","") if "unit" in r else res.get("unit","")
        summ   = r.get("summary","") if "summary" in r else res.get("summary","")
        conf   = r.get("confidence","?")
        error  = r.get("error","")
        icon   = "✓" if status == "ok" else "✗"
        lines.append(f"  ┌─ Step {step} [{expert}] {icon} {status.upper()} (confidence: {conf})")
        lines.append(f"  │  value_type : {vtype}")
        lines.append(f"  │  value      : {str(val)[:80]} {unit}")
        lines.append(f"  │  summary    : {summ[:100]}")
        if error:
            lines.append(f"  │  error      : {error}")
        lines.append(f"  └─")
    return "\n".join(lines)
