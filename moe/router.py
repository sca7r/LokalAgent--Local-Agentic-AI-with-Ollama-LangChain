"""
moe/router.py
Uses deepseek-r1 with constrained JSON output for routing.
Proper nested Pydantic models — $defs resolved via schema_utils.deref().
"""

import re, requests
from pydantic import BaseModel, Field
from typing import Literal
from moe.memory_manager import switch_to
from moe.config_loader import get_router_model, get_fallback_model, get_enabled_experts
from moe.expert_contract import call_with_schema


class RouterStep(BaseModel):
    step:        int = Field(description="Step number starting from 1")
    expert:      str = Field(description="Expert: web_search|math|code|file_read|file_write|api_call|doc_search|direct_answer")
    instruction: str = Field(description="Specific actionable instruction for this expert")


class RouterPlan(BaseModel):
    task_type: str = Field(description="Write exactly 'simple' if 1 step needed, or 'multi_step' if multiple steps needed")
    steps: list[RouterStep] = Field(
        description="Ordered list of steps. Minimum steps needed — never add steps not asked for."
    )


def _model_available(model: str) -> bool:
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        return any(model.split(":")[0] in m["name"] for m in res.json().get("models", []))
    except Exception:
        return False


def get_active_router_model() -> str:
    preferred = get_router_model()
    if _model_available(preferred):
        return preferred
    fallback = get_fallback_model()
    print(f"[Router] {preferred} not found — using {fallback}")
    return fallback


def route(task: str) -> dict:
    model = get_active_router_model()
    switch_to(model)

    experts     = get_enabled_experts()
    experts_str = "\n".join(f"- {k}: {v.get('description','')}" for k, v in experts.items())

    messages = [
        {
            "role": "system",
            "content": "You are a task router. Assign the minimum steps needed. Never add unrequested steps."
        },
        {
            "role": "user",
            "content":
                f"Available experts:\n{experts_str}\n\nTask: {task}\n\n"
                "Rules (strict):\n"
                "1. direct_answer for ALL knowledge questions — explanations, definitions, concepts, science, history.\n"
                "2. web_search ONLY for live/current data: prices, today's news, weather, recent events or if users asks to.\n"
                "3. math ONLY when user explicitly asks to calculate or compute a number.\n"
                "4. code ONLY when user explicitly asks to write or run code.\n"
                "5. file_write ONLY when user explicitly asks to save to a file.\n"
                "6. Use MINIMUM steps. Most tasks = 1 step.\n"
                "7. NEVER add steps the user did not ask for.\n\n"
                "Examples:\n"
                "  'explain quantum entanglement' → 1 step: direct_answer\n"
                "  'current gold price' → 1 step: web_search\n"
                "  'calculate 15% of 5000' → 1 step: math\n"
                "  'write a Python sort' → 1 step: code\n"
                "  'search gold price, calc 10%, save to file' → 3 steps: web_search, math, file_write"
        }
    ]

    plan = call_with_schema(model, messages, RouterPlan)

    if plan:
        steps = [{"step": s.step, "expert": s.expert, "instruction": s.instruction} for s in plan.steps]
        print(f"[Router] Task type: {plan.task_type} | Steps: {len(steps)}")
        for s in steps:
            print(f"[Router] Step {s['step']}: {s['expert']} — {s['instruction'][:60]}")
        return {"task_type": plan.task_type, "steps": steps, "model_used": model}

    print("[Router] Constrained call failed — fallback to direct_answer")
    return {
        "task_type": "simple",
        "steps": [{"step": 1, "expert": "direct_answer", "instruction": task}],
        "model_used": model,
    }
