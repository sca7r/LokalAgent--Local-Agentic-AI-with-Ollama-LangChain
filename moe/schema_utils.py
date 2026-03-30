"""
moe/schema_utils.py

Schema dereferencing utility.

WHY this exists:
Pydantic generates $defs + $ref for nested models — correct JSON Schema standard.
But Ollama's llama.cpp grammar engine cannot resolve $ref references (HTTP 500).
Production systems (OpenAI, Anthropic) resolve $refs server-side before passing
to their constrained decoder. We do the same here.

This lets us keep PROPER nested Pydantic models (like GPT/Claude use)
while still passing a valid inlined schema to Ollama.

The schema content is identical — we're just changing the representation,
not compromising the structure.
"""

import copy
import json
from typing import Any


def deref(schema: dict) -> dict:
    """
    Inline all $ref/$defs in a JSON Schema.

    Input (what Pydantic generates):
    {
      "properties": {
        "steps": {"items": {"$ref": "#/$defs/RouterStep"}}
      },
      "$defs": {
        "RouterStep": {"properties": {"step": ..., "expert": ..., "instruction": ...}}
      }
    }

    Output (what Ollama needs):
    {
      "properties": {
        "steps": {"items": {"properties": {"step": ..., "expert": ..., "instruction": ...}}}
      }
    }

    No information is lost — the schema is structurally identical,
    just with references resolved inline.
    """
    schema = copy.deepcopy(schema)
    defs   = schema.pop("$defs", {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                # Resolve the reference by inlining the definition
                ref_name = obj["$ref"].split("/")[-1]
                resolved = copy.deepcopy(defs.get(ref_name, {}))
                # Merge any sibling keys (like description) with the resolved def
                extra = {k: v for k, v in obj.items() if k != "$ref"}
                resolved.update(extra)
                return resolve(resolved)
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)


def safe_schema(model_class) -> dict:
    """
    Get a fully dereferenced JSON Schema from a Pydantic model.
    Safe to pass directly to Ollama format parameter.
    """
    return deref(model_class.model_json_schema())
