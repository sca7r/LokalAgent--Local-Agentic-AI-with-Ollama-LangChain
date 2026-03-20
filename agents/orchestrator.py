"""
agents/orchestrator.py
Orchestrator — coordinates the 3-agent pipeline:
  Planner → Executor → Reviewer → (loop if needed) → Final answer

This is the single entry point called by app.py for multi-agent mode.
"""

from agents.planner  import run_planner
from agents.executor import run_executor
from agents.reviewer import run_reviewer

MAX_ITERATIONS = 3


def run_multi_agent(task: str, model: str) -> dict:
    """
    Run the full Planner → Executor → Reviewer pipeline.

    Returns:
    {
        "final_answer": str,
        "iterations":   int,
        "plan":         list,
        "results":      list,
        "tools_used":   list[str],
    }
    """
    print(f"\n[Orchestrator] Starting multi-agent pipeline for: '{task}'")
    print(f"[Orchestrator] Model: {model} | Max iterations: {MAX_ITERATIONS}")

    feedback    = None
    tools_used  = []
    last_plan   = []
    last_results = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Orchestrator] ── Iteration {iteration}/{MAX_ITERATIONS} ──")

        # ── Step 1: Planner ───────────────────────────────────────────────
        print("[Orchestrator] Planner thinking...")
        plan = run_planner(task, model, feedback=feedback)
        print(f"[Orchestrator] Plan has {len(plan)} steps:")
        for step in plan:
            print(f"  Step {step.get('step')}: {step.get('description')} → {step.get('tool')}")

        last_plan = plan

        # ── Step 2: Executor ──────────────────────────────────────────────
        print("[Orchestrator] Executor running steps...")
        results = run_executor(plan, model=model)
        last_results = results

        # Collect all tools used
        for r in results:
            if r["tool"] != "none" and r["tool"] not in tools_used:
                tools_used.append(r["tool"])

        # ── Step 3: Reviewer ──────────────────────────────────────────────
        print("[Orchestrator] Reviewer evaluating...")
        review = run_reviewer(task, plan, results, model)
        print(f"[Orchestrator] Reviewer verdict: {review['verdict'].upper()}")

        if review["verdict"] == "pass":
            print("[Orchestrator] ✅ Task completed successfully!")
            return {
                "final_answer": review["final_answer"],
                "iterations":   iteration,
                "plan":         last_plan,
                "results":      last_results,
                "tools_used":   tools_used,
            }

        # Reviewer said FAIL — pass feedback to next iteration
        feedback = review["feedback"]
        print(f"[Orchestrator] ⚠️  Feedback: {feedback}")

    # Max iterations reached — return best available answer from results
    print("[Orchestrator] Max iterations reached — returning best result")
    best_output = "\n\n".join(
        f"Step {r['step']} ({r['tool']}): {r['output']}"
        for r in last_results if r.get("success") and r.get("output")
    )
    return {
        "final_answer": best_output or "Could not fully complete the task within the iteration limit.",
        "iterations":   MAX_ITERATIONS,
        "plan":         last_plan,
        "results":      last_results,
        "tools_used":   tools_used,
    }
