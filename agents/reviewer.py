"""
agents/reviewer.py
Reviewer Agent — evaluates execution results and decides:
  - PASS: results fully answer the user's task → format final answer
  - FAIL: results are incomplete → provide feedback to Planner for revision
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json


REVIEWER_SYSTEM_PROMPT = """You are a Reviewer agent evaluating task execution results.

Rules:
- If ALL steps succeeded and produced output, verdict = "pass"
- Only fail if a critical step produced NO output or an error
- Minor inaccuracies are acceptable — pass anyway
- Be lenient: the goal is to deliver value to the user

Return JSON only:
{
  "verdict": "pass" or "fail",
  "feedback": "only if fail: what specific step failed and why",
  "final_answer": "if pass: clear summary of what was accomplished with key results. if fail: empty string"
}

Return ONLY the JSON. Nothing else."""


def run_reviewer(task: str, plan: list, results: list, model: str) -> dict:
    """
    Review execution results.
    Returns {"verdict": "pass"|"fail", "feedback": str, "final_answer": str}
    """
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)

    # Build a readable summary of plan + results
    execution_summary = []
    for step, result in zip(plan, results):
        execution_summary.append(
            f"Step {result['step']}: {step.get('description', '')}\n"
            f"  Tool: {result['tool']}\n"
            f"  Input: {result['input'][:200]}\n"
            f"  Output: {result['output'][:500]}\n"
            f"  Success: {result['success']}"
        )

    summary_str = "\n\n".join(execution_summary)

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=f"Original task: {task}\n\nExecution results:\n{summary_str}"),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Extract JSON
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1:
        try:
            result = json.loads(content[start:end+1])
            # Ensure required keys exist
            return {
                "verdict":      result.get("verdict", "fail"),
                "feedback":     result.get("feedback", ""),
                "final_answer": result.get("final_answer", ""),
            }
        except json.JSONDecodeError:
            pass

    # Fallback — synthesise answer from successful results
    successful = [r for r in results if r.get("success")]
    if successful:
        answer = "\n\n".join(r["output"] for r in successful if r.get("output"))
        return {"verdict": "pass", "feedback": "", "final_answer": answer}
    return {"verdict": "fail", "feedback": "Could not complete the task.", "final_answer": ""}
