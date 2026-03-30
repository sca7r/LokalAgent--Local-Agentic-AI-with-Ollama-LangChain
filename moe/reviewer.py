"""
moe/reviewer.py
Uses deepseek-r1 with constrained JSON output for reviewing.
Proper nested Pydantic models — $defs resolved via schema_utils.deref().
"""

from pydantic import BaseModel, Field
from typing import Literal
from moe.memory_manager import switch_to
from moe.config_loader import get_router_model, get_fallback_model
from moe.expert_contract import call_with_schema
from moe.router import _model_available


class FixInstruction(BaseModel):
    step:            int = Field(description="Step number that needs fixing")
    expert:          str = Field(description="Expert to use for the fix")
    new_instruction: str = Field(description="Exact corrected instruction to fix the problem")


class ReviewResult(BaseModel):
    verdict: str = Field(description="Write exactly 'pass' if ALL steps succeeded with correct results, or 'fail' if anything went wrong")
    feedback: str = Field(
        default="",
        description="If fail: exactly what went wrong and why. If pass: empty string"
    )
    final_answer: str = Field(
        default="",
        description="If pass: one sentence summary of what was accomplished with key values. If fail: empty string"
    )
    fix_instructions: list[FixInstruction] = Field(description="If fail: specific fix for each broken step. Use empty list [] if verdict is pass")


def review(task: str, steps: list, results: list) -> dict:
    model = get_router_model() if _model_available(get_router_model()) else get_fallback_model()
    switch_to(model)

    summary = "\n\n".join(
        f"Step {r.get('step')} [{r.get('expert')}]\n"
        f"  Instruction: {s.get('instruction','')[:100]}\n"
        f"  Status     : {'ok' if (r.get('success') or r.get('status')=='ok') else 'error'}\n"
        f"  Value      : {r.get('value')} {r.get('unit','')}\n"
        f"  Summary    : {r.get('summary','')[:150]}"
        for s, r in zip(steps, results)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict reviewer. Evaluate whether the task was completed correctly. "
                "Pass only if all steps succeeded with correct, complete results."
            )
        },
        {
            "role": "user",
            "content": f"Task: {task}\n\nExecution results:\n{summary}"
        }
    ]

    result = call_with_schema(model, messages, ReviewResult)

    if result:
        print(f"[Reviewer] Verdict: {result.verdict.upper()}")
        if result.feedback:
            print(f"[Reviewer] Feedback: {result.feedback[:150]}")
        return {
            "verdict":          result.verdict,
            "feedback":         result.feedback,
            "final_answer":     result.final_answer,
            "fix_instructions": [
                {"step": f.step, "expert": f.expert, "new_instruction": f.new_instruction}
                for f in result.fix_instructions
            ],
        }

    # Fallback
    all_ok = all(r.get("success") or r.get("status") == "ok" for r in results)
    return {
        "verdict":          "pass" if all_ok else "fail",
        "feedback":         "" if all_ok else "Some steps failed",
        "final_answer":     results[-1].get("summary","") if all_ok and results else "",
        "fix_instructions": [],
    }
