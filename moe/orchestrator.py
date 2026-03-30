"""
moe/orchestrator.py
Full MoE pipeline with SSE progress streaming.

Key design decisions:
- router_model is defined once at the top of run_moe and passed everywhere
- context uses get_context_vars (contract-based) not _extract_value (regex)
- direct_answer bypasses MoE for pure knowledge questions
- _humanise generates user-facing answer from structured results
"""

import sys, os, re, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moe.router          import route, get_active_router_model
from moe.expert_contract import get_context_vars, format_for_terminal, validate, ok, err
from moe.reviewer        import review
from moe.memory_manager  import unload_current, switch_to
from moe.config_loader   import get_max_retries, is_direct_answer

from moe.experts import searcher, math_engine, file_expert, api_expert, coder, doc_expert


# ── Expert icons for UI progress ─────────────────────────────────────────────

EXPERT_ICONS = {
    "web_search":    "Searching web",
    "math":          "Calculating",
    "file_read":     "Reading file",
    "file_write":    "Writing file",
    "api_call":      "Calling API",
    "doc_search":    "Searching document",
    "code":          "Running code",
    "direct_answer": "Answering",
}


# ── Expert functions ──────────────────────────────────────────────────────────

def _direct_answer(instruction: str, model: str) -> dict:
    """
    Answer directly using deepseek from training knowledge.
    Used for: explanations, definitions, concepts, general knowledge.

    WHY separate from web_search:
    - Faster — no HTTP call, no extraction step
    - More reliable for factual/conceptual questions
    - deepseek-r1 trained on vast knowledge, doesn't need web for "explain X"
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        switch_to(model)
        llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0.3)
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer clearly and concisely."),
            HumanMessage(content=instruction),
        ]
        response = llm.invoke(messages)
        answer   = response.content.strip()
        # Strip deepseek-r1 thinking blocks
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        if not answer:
            answer = "I wasn't able to generate a response. Please try again."
        return ok(
            value      = answer,
            value_type = "text",
            unit       = "",
            summary    = answer[:120],
            expert     = "direct_answer",
            task       = instruction,
            raw        = answer,
            confidence = "high",
        )
    except Exception as e:
        return err(str(e), expert="direct_answer", task=instruction)


def _build_expert_map(model: str) -> dict:
    """
    Build the expert dispatch map.
    WHY function not constant: model is determined at runtime from config.
    """
    return {
        "web_search":    lambda i, ctx: searcher.run(i, ctx, model=model),
        "math":          lambda i, ctx: math_engine.run(i, ctx),
        "file_read":     lambda i, ctx: file_expert.read(i, ctx),
        "file_write":    lambda i, ctx: file_expert.write(i, ctx),
        "api_call":      lambda i, ctx: api_expert.run(i, ctx),
        "doc_search":    lambda i, ctx: doc_expert.run(i, ctx),
        "code":          lambda i, ctx: coder.run(i, ctx),
        "direct_answer": lambda i, ctx: _direct_answer(i, model),
    }


# ── Humanise: convert structured results to readable answer ──────────────────

def _humanise(results: list, task: str, model: str) -> str:
    """
    Ask deepseek to convert structured expert results to a readable answer.

    WHY separate step:
    - Experts communicate in typed JSON (for reliability between agents)
    - Users need natural language (not "step1_value: 4716.56")
    - Separating these concerns means each is done well
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        switch_to(model)
        llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0.3)

        summaries = "\n".join(
            f"- {r.get('expert','?')}: {r.get('summary', r.get('output',''))}"
            for r in results if r.get("success") or r.get("status") == "ok"
        )

        response = llm.invoke([
            SystemMessage(content="Convert these structured results into a clear, friendly answer for the user. Be concise. Do not mention steps or technical details."),
            HumanMessage(content=f"User asked: \"{task}\"\n\nResults:\n{summaries}"),
        ])
        answer = response.content.strip()
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer or summaries
    except Exception as e:
        print(f"[Orchestrator] _humanise failed: {e}")
        # Fallback — join summaries
        return "\n".join(
            r.get("summary", r.get("output", ""))
            for r in results if r.get("success") or r.get("status") == "ok"
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_moe(task: str, progress_callback=None) -> dict:
    """
    Full MoE pipeline:
    1. router  → plan (deepseek, constrained JSON)
    2. experts → results (each expert, typed contract)
    3. reviewer → verdict (deepseek, constrained JSON)
    4. humanise → user answer (deepseek, natural language)

    WHY router_model defined here:
    Previously it was referenced but never defined — caused NameError
    in every single expert call. Now set once, passed everywhere.
    """

    def emit(event_type: str, data: dict):
        if progress_callback:
            progress_callback({"type": event_type, **data})

    print(f"\n{'='*60}\n[MoE] Task: {task}\n{'='*60}")

    # Determine active model once — used by all LLM calls in this pipeline
    # WHY: prevents repeated model availability checks on every call
    router_model = get_active_router_model()
    expert_map   = _build_expert_map(router_model)

    # ── Shortcut for simple greetings ────────────────────────────────────────
    if is_direct_answer(task):
        emit("step", {"step": 1, "expert": "direct_answer", "status": "running", "label": "Answering"})
        task_lower = task.lower()
        if any(w in task_lower for w in ["name", "who are you", "introduce"]):
            answer = "I'm LokalAgent — a local agentic AI built with Ollama and a Mixture of Experts architecture."
        else:
            answer = "Hello! I'm LokalAgent. How can I help you?"
        emit("step", {"step": 1, "expert": "direct_answer", "status": "done", "output": answer})
        return {"final_answer": answer, "tools_used": [], "attempts": 1}

    all_tools  = []
    MAX_RETRIES = get_max_retries()

    # ── Route ────────────────────────────────────────────────────────────────
    emit("progress", {"label": "Router thinking...", "icon": "router"})
    plan  = route(task)
    steps = plan.get("steps", [])
    unload_current()  # free RAM — router done

    for attempt in range(1, MAX_RETRIES + 1):
        emit("progress", {"label": f"Attempt {attempt}/{MAX_RETRIES}", "icon": "attempt"})
        results      = []
        context_vars = {}

        for step in steps:
            snum        = step.get("step", "?")
            expert      = step.get("expert", "direct_answer")
            instruction = step.get("instruction", "")
            label       = EXPERT_ICONS.get(expert, expert)

            print(f"\n[Orchestrator] Step {snum}: {expert} — {instruction[:70]}")
            emit("step", {"step": snum, "expert": expert, "status": "running", "label": label})

            fn = expert_map.get(expert, expert_map["direct_answer"])

            try:
                result = fn(instruction, context_vars)
            except Exception as e:
                # WHY full contract in fallback: format_for_terminal needs all fields
                print(f"[Orchestrator] ✗ Step {snum} exception: {e}")
                result = err(str(e), expert=expert, task=instruction)

            # Stamp step metadata
            result["step"]   = snum
            result["expert"] = expert
            result["task"]   = instruction
            # Ensure output field for backward compat
            if "output" not in result:
                result["output"] = result.get("summary", result.get("error", ""))

            results.append(result)

            if expert not in all_tools:
                all_tools.append(expert)

            # Detect infrastructure failures (network, model not found)
            # WHY: these won't be fixed by retrying — stop the loop early
            is_infra_error = any(
                phrase in str(result.get("error", "")).lower()
                for phrase in ["name resolution", "connection", "timeout",
                               "model not found", "no such host", "network"]
            )
            if is_infra_error:
                print(f"[Orchestrator] ⚠ Infrastructure error in step {snum} — will not retry: {result.get('error','')}")
                result["_infra_error"] = True

            # Validate against contract schema
            is_valid, val_err = validate(result)
            if not is_valid:
                print(f"[Orchestrator] ⚠ Contract violation step {snum}: {val_err}")

            status = "done" if (result.get("success") or result.get("status") == "ok") else "error"
            emit("step", {
                "step":   snum,
                "expert": expert,
                "status": status,
                "output": result.get("summary", result.get("output", ""))[:150],
            })

            # Update context using contract — typed values, no regex
            # WHY get_context_vars not _extract_value:
            # result["value"] is already the correct typed value from the expert contract
            # _extract_value used regex on raw text and picked wrong numbers
            new_ctx = get_context_vars([result])
            context_vars.update(new_ctx)

            if new_ctx:
                key = f"step{snum}_value"
                print(f"[Orchestrator] Context: {key} = {context_vars.get(key)} "
                      f"[{context_vars.get(f'step{snum}_value_type','')}]")

        # Print full inter-expert log
        print(f"\n[MoE] Inter-expert message log (attempt {attempt}):")
        print(format_for_terminal(results))

        # Review
        emit("progress", {"label": "Reviewing results...", "icon": "review"})
        verdict = review(task, steps, results)
        unload_current()

        if verdict["verdict"] == "pass":
            emit("progress", {"label": "Complete!", "icon": "done"})
            answer = _humanise(results, task, router_model)
            unload_current()
            return {"final_answer": answer, "tools_used": all_tools, "attempts": attempt}

        feedback = verdict.get("feedback", "")
        print(f"\n[MoE] ❌ Attempt {attempt} failed: {feedback}")

        # Check if any step had a network/connectivity error
        # WHY: if web_search fails due to no internet, retrying will never help
        # Break early with a clear user message instead of looping pointlessly
        network_errors = [
            r for r in results
            if not (r.get("success") or r.get("status") == "ok")
            and any(w in str(r.get("error", "")).lower()
                    for w in ["name resolution", "connection", "network",
                               "timeout", "unavailable", "api_key not set",
                               "search failed"])
        ]
        if network_errors:
            experts_failed = [r["expert"] for r in network_errors]
            msg = (f"Web search is unavailable (no internet connection). "
                   f"I can still help with knowledge questions, calculations, "
                   f"and code — just don't ask for live data.")
            print(f"[MoE] Network error detected in: {experts_failed} — stopping retries")
            emit("progress", {"label": "Web search unavailable", "icon": "error"})
            return {"final_answer": msg, "tools_used": all_tools, "attempts": attempt}

        emit("progress", {"label": f"Retrying... {feedback[:50]}", "icon": "retry"})

        # Stop retrying if infrastructure error (network, model unavailable)
        # WHY: retrying won't fix a DNS failure or missing model
        has_infra_error = any(r.get("_infra_error") for r in results)
        if has_infra_error:
            print("[Orchestrator] Infrastructure error detected — skipping retries")
            best = "\n".join(
                r.get("summary", r.get("output", ""))
                for r in results
                if (r.get("success") or r.get("status") == "ok") and not r.get("_infra_error")
            )
            return {
                "final_answer": best or "Some steps could not complete due to network/infrastructure issues.",
                "tools_used":   all_tools,
                "attempts":     attempt,
            }

        # Apply specific fix instructions from reviewer
        fixes = {f["step"]: f for f in verdict.get("fix_instructions", [])}
        if fixes:
            steps = [
                {
                    "step":        s["step"],
                    "expert":      fixes[s["step"]]["expert"],
                    "instruction": fixes[s["step"]]["new_instruction"],
                } if s["step"] in fixes else s
                for s in steps
            ]
        else:
            # Re-route with failure context
            plan  = route(f"{task}\n\nPrevious attempt failed: {feedback}")
            steps = plan.get("steps", [])
            unload_current()

    # Max retries reached — return best available answer
    best = "\n".join(
        r.get("summary", r.get("output", ""))
        for r in results
        if r.get("success") or r.get("status") == "ok"
    )
    return {
        "final_answer": best or "Could not complete the task after maximum retries.",
        "tools_used":   all_tools,
        "attempts":     MAX_RETRIES,
    }
